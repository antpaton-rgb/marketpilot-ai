import os
from datetime import datetime, timezone, date
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_NAME = "MarketPilot AI v2"
PROVIDER = os.getenv("MARKET_DATA_PROVIDER", "twelvedata")
API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
SCAN_INTERVAL_SECONDS = 900
DEFAULT_RISK_PCT = 2.0
MAX_RISK_PCT = 3.0
MIN_REWARD_RISK = 1.5
DAILY_LOSS_LIMIT_PCT = 5.0
WEEKLY_LOSS_LIMIT_PCT = 10.0
DRAWDOWN_REVIEW_PCT = 20.0

app = FastAPI(title=APP_NAME, version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WATCHLIST = [
    {"symbol": "AAPL", "name": "Apple", "asset_type": "stock", "provider_symbol": "AAPL", "exchange": "NASDAQ", "quote_currency": "USD"},
    {"symbol": "MSFT", "name": "Microsoft", "asset_type": "stock", "provider_symbol": "MSFT", "exchange": "NASDAQ", "quote_currency": "USD"},
    {"symbol": "NVDA", "name": "NVIDIA", "asset_type": "stock", "provider_symbol": "NVDA", "exchange": "NASDAQ", "quote_currency": "USD"},
    {"symbol": "AMZN", "name": "Amazon", "asset_type": "stock", "provider_symbol": "AMZN", "exchange": "NASDAQ", "quote_currency": "USD"},
    {"symbol": "GOOGL", "name": "Alphabet", "asset_type": "stock", "provider_symbol": "GOOGL", "exchange": "NASDAQ", "quote_currency": "USD"},
    {"symbol": "META", "name": "Meta Platforms", "asset_type": "stock", "provider_symbol": "META", "exchange": "NASDAQ", "quote_currency": "USD"},
    {"symbol": "TSLA", "name": "Tesla", "asset_type": "stock", "provider_symbol": "TSLA", "exchange": "NASDAQ", "quote_currency": "USD"},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "asset_type": "etf", "provider_symbol": "SPY", "exchange": "NYSE ARCA", "quote_currency": "USD"},
    {"symbol": "QQQ", "name": "Invesco QQQ ETF", "asset_type": "etf", "provider_symbol": "QQQ", "exchange": "NASDAQ", "quote_currency": "USD"},
    {"symbol": "VWCE", "name": "Vanguard FTSE All-World UCITS ETF", "asset_type": "etf", "provider_symbol": "VWCE", "exchange": "XETRA", "quote_currency": "EUR"},
    {"symbol": "BTC/USD", "name": "Bitcoin", "asset_type": "crypto", "provider_symbol": "BTC/USD", "exchange": "CRYPTO", "quote_currency": "USD"},
    {"symbol": "ETH/USD", "name": "Ethereum", "asset_type": "crypto", "provider_symbol": "ETH/USD", "exchange": "CRYPTO", "quote_currency": "USD"},
    {"symbol": "SOL/USD", "name": "Solana", "asset_type": "crypto", "provider_symbol": "SOL/USD", "exchange": "CRYPTO", "quote_currency": "USD"},
    {"symbol": "XRP/USD", "name": "XRP", "asset_type": "crypto", "provider_symbol": "XRP/USD", "exchange": "CRYPTO", "quote_currency": "USD"},
    {"symbol": "BNB/USD", "name": "BNB", "asset_type": "crypto", "provider_symbol": "BNB/USD", "exchange": "CRYPTO", "quote_currency": "USD"},
]

LAST_SCAN = None
LAST_STATUS = "No market data configured"
LATEST_SIGNALS = []
# Closed paper-trade outcomes are recorded manually through the API. This is in-memory
# for this starter build; a database is required before deployment/production.
CLOSED_TRADES = []
PEAK_EQUITY_BY_CAPITAL = {}

class ScanRequest(BaseModel):
    capital_eur: float = Field(DEFAULT_CAPITAL_EUR, ge=100, le=10_000_000)
    risk_pct: float = Field(DEFAULT_RISK_PCT, ge=0.25, le=MAX_RISK_PCT)

class ClosedTradeRequest(BaseModel):
    pnl_eur: float
    capital_eur: float = Field(DEFAULT_CAPITAL_EUR, ge=100)
    note: Optional[str] = None

async def fetch_twelvedata(symbol: str, asset_type: str):
    if not API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY is not configured")
    params = {"symbol": symbol, "interval": "15min", "outputsize": 120, "apikey": API_KEY, "format": "JSON"}
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get("https://api.twelvedata.com/time_series", params=params)
        response.raise_for_status()
        data = response.json()
    if "values" not in data:
        raise RuntimeError(data.get("message", "Provider returned no values"))
    return [{"datetime": x["datetime"], "open": float(x["open"]), "high": float(x["high"]),
             "low": float(x["low"]), "close": float(x["close"]), "volume": float(x.get("volume") or 0)}
            for x in reversed(data["values"])]

def ema(values, period):
    alpha = 2 / (period + 1)
    result = []
    for value in values:
        result.append(value if not result else value * alpha + result[-1] * (1 - alpha))
    return result

def rsi(values, period=14):
    changes = [values[i] - values[i - 1] for i in range(len(values) - period, len(values))]
    gains = sum(max(change, 0) for change in changes) / period
    losses = sum(max(-change, 0) for change in changes) / period
    return 100.0 if losses == 0 else 100 - 100 / (1 + gains / losses)

def atr(rows, period=14):
    true_ranges = []
    for i in range(1, len(rows)):
        current, previous = rows[i], rows[i - 1]
        true_ranges.append(max(current["high"] - current["low"],
                               abs(current["high"] - previous["close"]),
                               abs(current["low"] - previous["close"])))
    return sum(true_ranges[-period:]) / min(period, len(true_ranges))

def risk_state(capital_eur: float):
    now = datetime.now(timezone.utc)
    today = now.date()
    week_start = today.fromordinal(today.toordinal() - today.weekday())
    daily_pnl = sum(t["pnl_eur"] for t in CLOSED_TRADES if t["date"] == today.isoformat())
    weekly_pnl = sum(t["pnl_eur"] for t in CLOSED_TRADES if t["date"] >= week_start.isoformat())
    total_pnl = sum(t["pnl_eur"] for t in CLOSED_TRADES)
    equity = capital_eur + total_pnl
    account_key = round(capital_eur, 2)
    peak_equity = max(PEAK_EQUITY_BY_CAPITAL.get(account_key, capital_eur), equity)
    PEAK_EQUITY_BY_CAPITAL[account_key] = peak_equity
    drawdown_pct = max(0.0, (peak_equity - equity) / peak_equity * 100) if peak_equity else 0.0
    daily_limit = capital_eur * DAILY_LOSS_LIMIT_PCT / 100
    weekly_limit = capital_eur * WEEKLY_LOSS_LIMIT_PCT / 100
    blocked_reasons = []
    if daily_pnl <= -daily_limit:
        blocked_reasons.append("Límite de pérdida diaria alcanzado")
    if weekly_pnl <= -weekly_limit:
        blocked_reasons.append("Límite de pérdida semanal alcanzado")
    if drawdown_pct >= DRAWDOWN_REVIEW_PCT:
        blocked_reasons.append("Drawdown del 10 %: estrategia bloqueada para revisión")
    return {"capital_eur": round(capital_eur, 2), "equity_eur": round(equity, 2),
            "daily_pnl_eur": round(daily_pnl, 2), "weekly_pnl_eur": round(weekly_pnl, 2),
            "drawdown_pct": round(drawdown_pct, 2), "daily_loss_limit_eur": round(daily_limit, 2),
            "weekly_loss_limit_eur": round(weekly_limit, 2), "trading_allowed": not blocked_reasons,
            "blocked_reasons": blocked_reasons}

def analyze(rows, capital_eur=DEFAULT_CAPITAL_EUR, risk_pct=DEFAULT_RISK_PCT):
    if len(rows) < 60:
        raise ValueError("Insufficient 15m candles")
    closes = [x["close"] for x in rows]
    e20, e50 = ema(closes, 20), ema(closes, 50)
    current_rsi = rsi(closes)
    macd_line = [a - b for a, b in zip(ema(closes, 12), ema(closes, 26))]
    macd_signal = ema(macd_line, 9)
    histogram = [a - b for a, b in zip(macd_line, macd_signal)]
    current_atr = atr(rows)
    current_volume = rows[-1]["volume"]
    prior_volumes = [x["volume"] for x in rows[-21:-1]]
    average_volume = sum(prior_volumes) / len(prior_volumes) if prior_volumes else 0
    volume_ok = current_volume > average_volume * 1.05 if average_volume > 0 else False

    trend_ok = e20[-1] > e50[-1] and closes[-1] > e20[-1]
    momentum_ok = 52 <= current_rsi <= 68
    macd_ok = histogram[-1] > 0 and histogram[-1] >= histogram[-2]
    volume_ratio = current_volume / average_volume if average_volume > 0 else None
    # A small positive breakout confirmation avoids buying solely on lagging indicators.
    breakout_ok = closes[-1] >= max(x["high"] for x in rows[-6:-1])

    score = (30 if trend_ok else 0) + (25 if momentum_ok else 0) + (20 if macd_ok else 0) + (15 if volume_ok else 0) + (10 if breakout_ok else 0)
    entry = closes[-1]
    stop = entry - 1.5 * current_atr
    target = entry + (entry - stop) * MIN_REWARD_RISK
    risk_amount = capital_eur * risk_pct / 100
    stop_distance = entry - stop
    # Never fabricate EUR sizing for USD-quoted assets: FX conversion is not wired yet.
    quote_currency = rows[-1].get("quote_currency", "")
    quantity = risk_amount / stop_distance if stop_distance > 0 and quote_currency == "EUR" else None
    position_value = quantity * entry if quantity is not None else None
    actual_risk = quantity * stop_distance if quantity is not None else None
    rr = (target - entry) / stop_distance if stop_distance > 0 else 0

    gates = []
    if not trend_ok: gates.append("tendencia no alcista")
    if not momentum_ok: gates.append("RSI fuera de zona 52–68")
    if not macd_ok: gates.append("MACD sin confirmación alcista")
    if not volume_ok: gates.append("volumen insuficiente")
    if not breakout_ok: gates.append("sin ruptura del máximo reciente")
    if rr < MIN_REWARD_RISK: gates.append("ratio beneficio/riesgo insuficiente")

    # BUY is deliberately selective: strong score plus mandatory trend, momentum and MACD.
    buy = score >= 80 and trend_ok and momentum_ok and macd_ok and rr >= MIN_REWARD_RISK
    status = "BUY" if buy else ("WATCH" if score >= 55 and trend_ok else "PASS")
    reasons = [name for name, passed in [("tendencia alcista", trend_ok), ("momentum equilibrado", momentum_ok),
                ("MACD confirmado", macd_ok), ("volumen confirmado", volume_ok), ("ruptura reciente", breakout_ok)] if passed]
    return {"score": score, "status": status, "direction": "LONG_ONLY", "close": round(entry, 8),
            "rsi": round(current_rsi, 2), "atr": round(current_atr, 8),
            "entry": round(entry, 8), "stop_loss": round(stop, 8), "take_profit": round(target, 8),
            "reward_risk": round(rr, 2), "risk_pct": risk_pct, "risk_eur": round(risk_amount, 2),
            "quote_currency": quote_currency or "PROVIDER_QUOTE",
            "quantity": round(quantity, 8) if quantity is not None else None,
            "position_value_eur": round(position_value, 2) if position_value is not None else None,
            "sizing_note": "Tamaño calculado en EUR" if quantity is not None else "Falta conversión de divisa para dimensionar en EUR",
            "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
            "reason": ", ".join(reasons) if reasons else "No supera los filtros long",
            "failed_filters": gates}

@app.get("/health")
async def health():
    return {"app": APP_NAME, "version": "2.1.0", "provider": PROVIDER,
            "provider_key_configured": bool(API_KEY), "last_scan": LAST_SCAN,
            "data_status": LAST_STATUS, "scan_interval_seconds": SCAN_INTERVAL_SECONDS,
            "strategy": "LONG_ONLY", "objective": "BALANCED", "risk_pct_default": DEFAULT_RISK_PCT,
            "max_risk_pct": MAX_RISK_PCT}

@app.get("/api/v1/assets")
async def assets():
    return WATCHLIST[:100]

@app.get("/api/v1/signals")
async def signals():
    return LATEST_SIGNALS

@app.get("/api/v1/risk")
async def get_risk(capital_eur: float = DEFAULT_CAPITAL_EUR):
    return risk_state(capital_eur)

@app.post("/api/v1/paper-trades/close")
async def record_closed_trade(trade: ClosedTradeRequest):
    CLOSED_TRADES.append({"date": datetime.now(timezone.utc).date().isoformat(),
                          "pnl_eur": float(trade.pnl_eur), "note": (trade.note or "")[:200]})
    state = risk_state(trade.capital_eur)
    return {"recorded": True, "risk": state}

@app.post("/api/v1/scan")
async def scan(config: ScanRequest = ScanRequest()):
    global LAST_SCAN, LAST_STATUS, LATEST_SIGNALS
    if PROVIDER != "twelvedata":
        raise HTTPException(400, "This starter build currently implements Twelve Data only")
    if not API_KEY:
        raise HTTPException(503, "Set TWELVEDATA_API_KEY on the server")
    state = risk_state(config.capital_eur)
    results = []
    for asset in WATCHLIST:
        try:
            rows = await fetch_twelvedata(asset["provider_symbol"], asset["asset_type"])
            rows[-1]["quote_currency"] = asset.get("quote_currency", "USD")
            signal = analyze(rows, config.capital_eur, config.risk_pct)
            if not state["trading_allowed"] and signal["status"] == "BUY":
                signal["status"] = "BLOCKED_RISK"
                signal["failed_filters"].extend(state["blocked_reasons"])
            results.append({**signal, "symbol": asset["symbol"], "asset_type": asset["asset_type"],
                            "as_of": rows[-1]["datetime"]})
        except Exception as error:
            results.append({"symbol": asset["symbol"], "asset_type": asset["asset_type"],
                            "status": "NO_DATA", "direction": "LONG_ONLY", "score": 0,
                            "close": None, "reason": str(error)[:180]})
    LATEST_SIGNALS = results
    LAST_SCAN = datetime.now(timezone.utc).isoformat()
    good = sum(1 for item in results if item["status"] != "NO_DATA")
    LAST_STATUS = f"{good}/{len(results)} activos con datos"
    return {"last_scan": LAST_SCAN, "data_status": LAST_STATUS, "risk": state, "signals": results}
