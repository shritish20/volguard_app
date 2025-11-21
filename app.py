import os
from datetime import datetime
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load .env variables (if present)
load_dotenv()

# Import backend
import volguard_backend as vg

# ============================================================
# INIT
# ============================================================

def init_volguard():
    if st.session_state.get("vg_initialized"):
        return

    analytics = vg.RobustRealDataAnalytics()
    vol_surface = vg.RobustVolatilitySurface2D()
    mdm = vg.InstitutionalMarketData(vg.UPSTOX_ACCESS_TOKEN, analytics, vol_surface)
    risk = vg.InstitutionalRiskManager(vg.ACCOUNT_SIZE, analytics)
    exec_engine = vg.InstitutionalExecutionEngine(mdm, risk)
    strategy_engine = vg.AdvancedStrategyEngine(mdm)

    st.session_state.vg_initialized = True
    st.session_state.mdm = mdm
    st.session_state.analytics = analytics
    st.session_state.vol_surface = vol_surface
    st.session_state.risk = risk
    st.session_state.exec = exec_engine
    st.session_state.strategy_engine = strategy_engine

    st.session_state.symbol = "NIFTY 50"
    st.session_state.last_metrics = None
    st.session_state.last_strategy = None
    st.session_state.last_chain = None
    st.session_state.open_trades = []


def get_components():
    return (
        st.session_state.mdm,
        st.session_state.analytics,
        st.session_state.vol_surface,
        st.session_state.risk,
        st.session_state.exec,
        st.session_state.strategy_engine,
    )

# ============================================================
# MARKET & STRATEGY
# ============================================================

def refresh_market():
    mdm, analytics, vol_surface, risk, exec_engine, engine = get_components()

    spot, vix = mdm.get_enhanced_spot_vix()
    expiry = mdm.get_weekly_expiry()
    chain = mdm.get_option_chain_with_validation(st.session_state.symbol, expiry)

    if not chain:
        st.warning("Option chain empty or invalid.")
        return None

    engine.update_metrics(spot, vix, chain, analytics)
    strategy_name, legs = engine.select_strategy()

    st.session_state.last_metrics = engine.metrics
    st.session_state.last_strategy = {
        "name": strategy_name,
        "legs": legs,
        "expiry": expiry,
        "spot": spot,
        "vix": vix,
    }
    st.session_state.last_chain = chain

    return True

# ============================================================
# TRADE CREATION
# ============================================================

def build_trade_from_legs(legs, expiry, lots, strategy_name="CUSTOM"):
    mdm, analytics, vol_surface, risk, exec_engine, engine = get_components()
    symbol = st.session_state.symbol
    spot, _ = mdm.get_enhanced_spot_vix()

    positions = []
    net_premium = 0
    vega_total = 0

    for leg in legs:
        strike = float(leg["strike"])
        type_ = leg["type"].upper()
        side = leg["side"].upper()

        qty_sign = -1 if side == "SELL" else 1
        qty = qty_sign * lots * vg.LOT_SIZE

        key = mdm.get_option_instrument_key(symbol, expiry, strike, type_)
        quote = mdm.api.get_market_quote_ltp([key])
        ltp = float(
            quote.get("data", {}).get(key, {}).get("last_price", 50) or 50
        )

        try:
            g = mdm.pricing_engine.calculate_greeks(spot, strike, type_, expiry)
        except:
            g = vg.GreeksSnapshot(
                timestamp=datetime.now(vg.IST),
                total_delta=0,
                total_gamma=0,
                total_theta=0,
                total_vega=0,
                total_rho=0,
                staleness_seconds=0,
            )

        pos = vg.Position(
            symbol=symbol,
            instrument_key=key,
            strike=strike,
            option_type=type_,
            quantity=qty,
            entry_price=ltp,
            entry_time=datetime.now(vg.IST),
            current_price=ltp,
            current_greeks=g,
        )

        positions.append(pos)
        net_premium += (-qty_sign) * ltp
        vega_total += g.total_vega * qty

    trade = vg.MultiLegTrade(
        legs=positions,
        strategy_type=strategy_name,
        net_premium_per_share=net_premium,
        entry_time=datetime.now(vg.IST),
        lots=lots,
        trading_mode=vg.TradingMode.SYSTEMATIC,
        expiry_date=expiry,
        trade_vega=vega_total,
    )
    return trade

# ============================================================
# UI PANELS
# ============================================================

def sidebar():
    st.sidebar.title("⚙️ VolGuard v8.1")

    if vg.PAPER_TRADING:
        st.sidebar.success("PAPER MODE")
    else:
        st.sidebar.error("LIVE MODE")

    st.sidebar.text_input("Underlying Symbol", key="symbol")

    exec_engine = st.session_state.exec
    ks = exec_engine.kill_switch.get_status()

    st.sidebar.markdown("### Kill Switch")
    st.sidebar.write(f"Active: {ks['activated']}")
    if ks["activated"]:
        st.sidebar.write(f"Reason: {ks['activation_reason']}")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Activate KS"):
        exec_engine.kill_switch.activate("UI Activation")
    if col2.button("Deactivate KS"):
        exec_engine.kill_switch.deactivate()

def dashboard():
    st.subheader("📊 Market Dashboard")

    if st.button("Refresh Live Market Data"):
        if refresh_market():
            st.success("Updated.")

    m = st.session_state.last_metrics
    s = st.session_state.last_strategy

    if not m:
        st.info("Click refresh first.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Spot", f"{m['spot']:.2f}")
    col1.metric("VIX", f"{m['vix']:.2f}")
    col1.metric("IVP", f"{m['ivp']:.1f}%")

    col2.metric("PCR", f"{m['pcr']:.2f}")
    col2.metric("Skew", f"{m['skew']:.2f}")
    col2.metric("IV-RV", f"{m['iv_rv_spread']:.2f}")

    col3.metric("Regime", m["regime"])
    col3.metric("Sub-Regime", m["sub_regime"])
    col3.metric("Confidence", f"{m['regime_confidence']*100:.1f}%")

    st.markdown("---")
    st.subheader("🧠 Strategy Suggestion")

    if s and s["name"] != "SKIP":
        st.write(f"**{s['name']}**")
        st.dataframe(pd.DataFrame(s["legs"]))
    else:
        st.warning("No trade recommended.")

def vol_surface_tab():
    st.subheader("🌡️ Volatility Surface")

    chain = st.session_state.last_chain
    if not chain:
        st.info("Refresh market first.")
        return

    vs = st.session_state.vol_surface

    expiries = vs.expiries
    if not expiries:
        st.warning("No calibrated expiries.")
        return

    expiry = st.selectbox("Select Expiry", expiries)

    strikes = []
    ivs = []

    for d in chain.get("data", {}).values():
        if d.get("expiry") == expiry:
            strike = float(d.get("strike", 0))
            if strike > 0:
                strikes.append(strike)
                ivs.append(vs.get_volatility(strike, expiry))

    if strikes:
        df = pd.DataFrame({"Strike": strikes, "IV": ivs}).sort_values("Strike")
        st.line_chart(df.set_index("Strike"))
        st.dataframe(df)
    else:
        st.warning("No strikes found for expiry.")

def strategy_tab():
    st.subheader("🧠 Strategy Details")

    s = st.session_state.last_strategy
    m = st.session_state.last_metrics

    if not s or not m:
        st.info("Refresh dashboard first.")
        return

    st.write(f"**Strategy:** {s['name']} | Expiry: {s['expiry']}")

    st.dataframe(pd.DataFrame(s["legs"]))

    st.markdown("### Regime History (Recent)")
    try:
        eng = st.session_state.strategy_engine
        rh = list(eng.regime_detector.regime_history)
        if rh:
            df = pd.DataFrame(rh)
            df["timestamp"] = df["timestamp"].astype(str)
            st.dataframe(df.tail(20))
        else:
            st.info("No regime history yet.")
    except:
        pass

def execution_tab():
    st.subheader("⚙️ Execution")

    tabs = st.tabs(["Engine Strategy", "Manual Builder"])

    # Engine strategy
    with tabs[0]:
        s = st.session_state.last_strategy
        if not s or s["name"] == "SKIP":
            st.info("No engine strategy yet.")
        else:
            st.write(f"Strategy: {s['name']} | Expiry: {s['expiry']}")
            st.dataframe(pd.DataFrame(s["legs"]))

            lots = st.number_input("Lots", 1, 20, 1)
            if st.button("Open Engine Strategy"):
                trade = build_trade_from_legs(s["legs"], s["expiry"], lots, s["name"])
                res = st.session_state.exec.place_open_trade(trade)
                if res:
                    st.session_state.open_trades.append(trade)
                    st.success("Trade opened.")

    # Manual builder
    with tabs[1]:
        default = [{"side":"SELL","type":"CE","strike":0.0},
                   {"side":"BUY","type":"CE","strike":0.0},
                   {"side":"SELL","type":"PE","strike":0.0},
                   {"side":"BUY","type":"PE","strike":0.0}]

        df = st.data_editor(pd.DataFrame(default), key="manual")
        expiry = st.text_input("Expiry (YYYY-MM-DD)", value=s["expiry"] if s else "")
        lots = st.number_input("Lots", 1, 20, 1)

        if st.button("Open Manual Trade"):
            legs = []
            for _, r in df.iterrows():
                if float(r["strike"]) > 0:
                    legs.append({"side": r["side"], "type": r["type"], "strike": float(r["strike"])})
            if not legs:
                st.error("No valid legs.")
            else:
                trade = build_trade_from_legs(legs, expiry, lots)
                res = st.session_state.exec.place_open_trade(trade)
                if res:
                    st.session_state.open_trades.append(trade)
                    st.success("Manual trade opened.")

    st.markdown("---")
    st.subheader("Current Session Trades")
    rows = []
    for t in st.session_state.open_trades:
        rows.append({
            "Strategy": t.strategy_type,
            "Lots": t.lots,
            "Expiry": t.expiry_date,
            "Status": t.status.value,
            "Net Premium": t.net_premium_per_share,
            "Unrealized PnL": t.total_unrealized_pnl(),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("No trades yet.")

def risk_tab():
    st.subheader("🛡️ Risk")

    mdm, analytics, vol_surface, risk, exec_engine, engine = get_components()

    open_pnl = sum(t.total_unrealized_pnl() for t in st.session_state.open_trades)
    risk.update_equity(open_pnl)
    dd = risk.drawdown()
    pv = sum(t.trade_vega for t in st.session_state.open_trades)

    spot, vix = mdm.get_enhanced_spot_vix()
    var, es = risk.calculate_monte_carlo_var(pv, vix)

    col1, col2, col3 = st.columns(3)
    col1.metric("Equity", f"{risk.equity_now:,.0f}")
    col1.metric("Open PnL", f"{open_pnl:,.0f}")

    col2.metric("Drawdown", f"{dd*100:.2f}%")
    col2.metric("Portfolio Vega", f"{pv:.2f}")

    col3.metric("99% VaR", f"{var:,.0f}")
    col3.metric("99% ES", f"{es:,.0f}")

def journal_tab():
    st.subheader("📘 Trade Journal")

    file = getattr(vg, "TRADE_JOURNAL_FILE", "volguard_trade_journal_v8_1.csv")

    if not os.path.isfile(file):
        st.info("No journal file found yet.")
        return

    df = pd.read_csv(file)
    st.dataframe(df)

    if "realized_pnl" in df.columns:
        daily = (
            df.groupby(df["timestamp"].str[:10])["realized_pnl"]
            .sum()
            .reset_index()
            .rename(columns={"timestamp":"date"})
        )
        if not daily.empty:
            st.line_chart(daily.set_index("date")["realized_pnl"])

def health_tab():
    st.subheader("💓 System Health")

    mdm, analytics, vol_surface, risk, exec_engine, engine = get_components()

    if st.button("Run Health Check"):
        result = exec_engine.health_monitor.run_health_checks({
            "market_data": mdm,
            "api": mdm.api,
            "pricing_engine": mdm.pricing_engine,
        })
        if result:
            st.success("System Healthy")
        else:
            st.error("Some issues in components")

    st.json(exec_engine.health_monitor.get_health_report())

def settings_tab():
    st.subheader("⚙️ Settings")

    risk = st.session_state.risk
    new_size = st.number_input(
        "Risk Manager Session Account Size",
        min_value=100000.0,
        max_value=10000000.0,
        value=float(risk.account_size),
        step=50000.0,
    )

    if st.button("Update Account Size"):
        risk.account_size = new_size
        st.success("Updated session risk size.")

# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(page_title="VolGuard v8.1 Terminal", layout="wide")

    st.title("🛡️ VolGuard v8.1 — Streamlit Terminal")
    st.caption("Soft-Real Institutional Trading System")

    init_volguard()
    sidebar()

    tabs = st.tabs([
        "📊 Dashboard",
        "🌡️ Vol Surface",
        "🧠 Strategy",
        "⚙️ Execution",
        "🛡️ Risk",
        "📘 Journal",
        "💓 Health",
        "⚙️ Settings",
    ])

    with tabs[0]: dashboard()
    with tabs[1]: vol_surface_tab()
    with tabs[2]: strategy_tab()
    with tabs[3]: execution_tab()
    with tabs[4]: risk_tab()
    with tabs[5]: journal_tab()
    with tabs[6]: health_tab()
    with tabs[7]: settings_tab()


if __name__ == "__main__":
    main()
