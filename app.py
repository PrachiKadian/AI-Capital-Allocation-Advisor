import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import ta
from scipy.optimize import minimize
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
import json
import os
import hashlib

# ==========================================
# 0. SYSTEM CONFIGURATION
# ==========================================
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Capital Allotment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Authentication System ---
USER_DB_FILE = "users_db.json"

def load_users():
    if not os.path.exists(USER_DB_FILE): return {}
    with open(USER_DB_FILE, "r") as f: return json.load(f)

def save_users(users):
    with open(USER_DB_FILE, "w") as f: json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def auth_screen():
    st.title("AI Capital Allotment")
    st.markdown("### Secure Access Portal")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Secure Login"):
            db = load_users()
            if u in db and db[u] == hash_password(p):
                st.session_state["authenticated"] = True
                st.session_state["username"] = u
                st.rerun()
            else:
                st.error("Authentication Failed.")
    
    with tab2:
        nu = st.text_input("New Username", key="reg_user")
        np_pass = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Create Profile"):
            db = load_users()
            if nu in db:
                st.error("User exists.")
            else:
                db[nu] = hash_password(np_pass)
                save_users(db)
                st.success("Profile created. Please login.")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    auth_screen()
    st.stop()

# ==========================================
# 1. ASSET CONFIGURATION
# ==========================================
RISK_FREE_RATE = 0.070 

ASSET_MAP = {
    # Blue Chip Stocks
    'RELIANCE.NS': 'Reliance Industries', 'TCS.NS': 'TCS', 'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys', 'ICICIBANK.NS': 'ICICI Bank', 'ITC.NS': 'ITC Ltd',
    'SBIN.NS': 'SBI', 'BHARTIARTL.NS': 'Bharti Airtel', 'LT.NS': 'Larsen & Toubro',
    'HINDUNILVR.NS': 'HUL', 'AXISBANK.NS': 'Axis Bank', 'MARUTI.NS': 'Maruti Suzuki',
    'SUNPHARMA.NS': 'Sun Pharma', 'BAJFINANCE.NS': 'Bajaj Finance', 'TATAMOTORS.NS': 'Tata Motors',
    'ADANIENT.NS': 'Adani Enterprises', 'TITAN.NS': 'Titan Company', 'WIPRO.NS': 'Wipro',
    'POWERGRID.NS': 'Power Grid', 'NTPC.NS': 'NTPC',
    
    # Mutual Funds (Direct Growth Proxies)
    '0P0000XVAA.BO': 'HDFC Top 100', '0P0000XW8F.BO': 'SBI Bluechip',
    '0P0000XVK5.BO': 'Axis Bluechip', '0P0000XVTR.BO': 'ICICI Pru Value',
    '0P00005WLZ.BO': 'Nippon Small Cap', '0P00009J3J.BO': 'Mirae Large Cap'
}

STOCKS = [k for k in ASSET_MAP.keys() if '.NS' in k]
FUNDS = [k for k in ASSET_MAP.keys() if '.BO' in k]

# ==========================================
# 2. DATA ENGINE
# ==========================================
def fetch_index_data(ticker):
    """Robust Index Fetcher (5-day lookback)."""
    try:
        idx = yf.Ticker(ticker)
        hist = idx.history(period="5d")
        if hist.empty: return 0.0, 0.0
        
        curr = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2] if len(hist) > 1 else curr
        change_pct = ((curr - prev) / prev) * 100
        return curr, change_pct
    except: return 0.0, 0.0

@st.cache_data(ttl=60)
def fetch_live_price(ticker):
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if not data.empty:
            price = data['Close'].iloc[-1]
            return float(price.item()) if hasattr(price, 'item') else float(price)
        data_d = yf.download(ticker, period="1d", progress=False)
        return float(data_d['Close'].iloc[-1])
    except: return 0.0

@st.cache_data(ttl=3600)
def fetch_historical_matrix(tickers, period="2y"):
    """Fetch clean Close price matrix."""
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, progress=False)
        prices = pd.DataFrame()
        for t in tickers:
            try:
                if len(tickers) == 1:
                    series = data['Close'] if 'Close' in data.columns else data
                else:
                    series = data[t]['Close'] if t in data.columns.levels[0] else None
                if series is not None: prices[t] = series
            except: pass
        prices.ffill(inplace=True)
        prices.dropna(how='any', inplace=True)
        return prices
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_analysis_data(ticker, period="1y"):
    """Deep fetch with variable period."""
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            try: data = data.xs(ticker, axis=1, level=0)
            except: data.columns = data.columns.get_level_values(0)
        return data
    except: return pd.DataFrame()

# ==========================================
# 3. ANALYTICS CORE
# ==========================================
def calculate_market_movers():
    """Calculates top gainers/losers based on 7-day history (avoids weekend gaps)."""
    df = fetch_historical_matrix(STOCKS, period="7d")
    if df.empty or len(df) < 2: return pd.DataFrame(), pd.DataFrame()
    
    try:
        # Get last valid close and the close 1 day prior
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        pct_change = ((latest - prev) / prev) * 100
        
        top_gainers = pct_change.sort_values(ascending=False).head(5)
        top_losers = pct_change.sort_values(ascending=True).head(5)
        return top_gainers, top_losers
    except:
        return pd.DataFrame(), pd.DataFrame()

def optimize_portfolio(tickers, risk_level):
    df = fetch_historical_matrix(tickers, "1y")
    if df.empty or len(df.columns) < 2: return [], []
    
    returns = df.pct_change().dropna()
    mu = returns.mean() * 252
    sigma = returns.cov() * 252
    n_assets = len(df.columns)
    
    # RISK ALLOTMENT LOGIC
    # Conservative: Restricts max allocation to 15% per asset, forces diversification
    # Aggressive: Allows up to 40% per asset, allows concentration
    if risk_level == "Conservative":
        bounds = tuple((0.0, 0.15) for _ in range(n_assets))
    elif risk_level == "Aggressive":
        bounds = tuple((0.0, 0.40) for _ in range(n_assets))
    else: # Moderate
        bounds = tuple((0.0, 0.25) for _ in range(n_assets))
        
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    def neg_sharpe(w):
        return -((np.sum(w * mu) - RISK_FREE_RATE) / np.sqrt(np.dot(w.T, np.dot(sigma, w))))
    
    try:
        res = minimize(neg_sharpe, [1./n_assets]*n_assets, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x, df.columns
    except: return [], []

# ==========================================
# 4. DASHBOARD UI
# ==========================================
def main():
    # --- Sidebar ---
    st.sidebar.title("Configuration")
    st.sidebar.write(f"User: **{st.session_state['username']}**")
    
    capital = st.sidebar.number_input("Capital (₹)", 50000, 100000000, 500000, step=100000)
    risk_profile = st.sidebar.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
    
    if st.sidebar.button("Secure Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

    st.title("AI Capital Allotment System")

    tabs = st.tabs(["Market Dashboard", "Technical Analysis", "Mutual Funds", "Allocation Engine", "Executive Report"])
    
    # --- TAB 1: DASHBOARD ---
    with tabs[0]:
        st.subheader("Market Indices")
        c1, c2, c3 = st.columns(3)
        n_val, n_pct = fetch_index_data("^NSEI")
        b_val, b_pct = fetch_index_data("^NSEBANK")
        s_val, s_pct = fetch_index_data("^BSESN")
        
        c1.metric("NIFTY 50", f"₹{n_val:,.0f}", f"{n_pct:.2f}%")
        c2.metric("BANK NIFTY", f"₹{b_val:,.0f}", f"{b_pct:.2f}%")
        c3.metric("SENSEX", f"₹{s_val:,.0f}", f"{s_pct:.2f}%")
        
        st.divider()
        st.subheader("Daily Market Movers")
        gainers, losers = calculate_market_movers()
        
        if not gainers.empty:
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("#### Top Gainers")
                g_df = pd.DataFrame({"Asset": [ASSET_MAP.get(x,x) for x in gainers.index], "Change": [f"+{x:.2f}%" for x in gainers.values]})
                st.table(g_df)
            with mc2:
                st.markdown("#### Top Losers")
                l_df = pd.DataFrame({"Asset": [ASSET_MAP.get(x,x) for x in losers.index], "Change": [f"{x:.2f}%" for x in losers.values]})
                st.table(l_df)
        else:
            st.warning("Market data is syncing. Please wait for the API to respond.")

    # --- TAB 2: TECHNICAL ANALYSIS ---
    with tabs[1]:
        st.subheader("Advanced Stock Analysis")
        col_ctrl, col_disp = st.columns([1, 3])
        
        with col_ctrl:
            selected_stock = st.selectbox("Select Asset", STOCKS, format_func=lambda x: ASSET_MAP[x])
            time_frame = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
            if st.button("Refresh Data"): st.cache_data.clear()
            
        with col_disp:
            df_stock = fetch_analysis_data(selected_stock, period=time_frame)
            if not df_stock.empty:
                curr_price = fetch_live_price(selected_stock)
                st.metric(f"{ASSET_MAP[selected_stock]} ({time_frame})", f"₹{curr_price:,.2f}")
                
                # Indicators
                df_stock['SMA_20'] = ta.trend.sma_indicator(df_stock['Close'], window=20)
                df_stock['SMA_50'] = ta.trend.sma_indicator(df_stock['Close'], window=50)
                bb = ta.volatility.BollingerBands(df_stock['Close'], window=20, window_dev=2)
                df_stock['BB_H'] = bb.bollinger_hband()
                df_stock['BB_L'] = bb.bollinger_lband()
                
                # AI Trend
                df_stock['idx'] = np.arange(len(df_stock))
                clean = df_stock.dropna(subset=['Close'])
                reg = LinearRegression().fit(clean[['idx']], clean['Close'])
                future_idx = np.arange(clean['idx'].iloc[-1], clean['idx'].iloc[-1] + 30).reshape(-1, 1)
                forecast = reg.predict(future_idx)
                future_dates = [clean.index[-1] + timedelta(days=i) for i in range(1, 31)]

                # Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_stock.index, open=df_stock['Open'], high=df_stock['High'],
                                low=df_stock['Low'], close=df_stock['Close'], name="Price"))
                fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['SMA_50'], line=dict(color='orange'), name="SMA 50"))
                fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['BB_H'], line=dict(color='gray', dash='dot'), name="BB High"))
                fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['BB_L'], line=dict(color='gray', dash='dot'), name="BB Low"))
                fig.add_trace(go.Scatter(x=future_dates, y=forecast, line=dict(color='blue', dash='dash'), name="AI Forecast"))
                
                fig.update_layout(height=500, xaxis_rangeslider_visible=False, title="Price Action & AI Projection")
                st.plotly_chart(fig, use_container_width=True)
                
                # Fundamentals
                with st.expander("Fundamental Data"):
                    try:
                        info = yf.Ticker(selected_stock).info
                        f1, f2, f3 = st.columns(3)
                        f1.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
                        f2.metric("Market Cap", f"₹{info.get('marketCap', 0)/1e9:.1f}B")
                        f3.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%")
                    except: st.write("Fundamental data unavailable.")

    # --- TAB 3: MUTUAL FUNDS ---
    with tabs[2]:
        st.subheader("Top Performing Mutual Funds")
        mf_df = fetch_historical_matrix(FUNDS, "1y")
        
        if not mf_df.empty:
            # Calculate Returns for List
            mf_rets = mf_df.pct_change().sum() * 100
            mf_list = pd.DataFrame({
                "Fund Name": [ASSET_MAP.get(x,x) for x in mf_rets.index],
                "Ticker": mf_rets.index,
                "1Y Return (%)": mf_rets.values
            }).sort_values("1Y Return (%)", ascending=False)
            
            st.dataframe(mf_list.style.format({"1Y Return (%)": "{:.2f}%"}), use_container_width=True)
            
            st.subheader("Performance Comparison (Base=100)")
            norm_mf = mf_df / mf_df.iloc[0] * 100
            norm_mf.columns = [ASSET_MAP.get(c,c) for c in norm_mf.columns]
            st.line_chart(norm_mf)

    # --- TAB 4: ALLOCATION ENGINE ---
    with tabs[3]:
        st.subheader("AI Allocation Engine")
        st.info(f"Optimization Mode: **{risk_profile}** (Allocations adjusted for risk tolerance)")
        
        c1, c2 = st.columns(2)
        with c1:
            univ_stock = st.multiselect("Select Stocks for Portfolio", STOCKS, default=STOCKS[:5], format_func=lambda x: ASSET_MAP[x])
        with c2:
            univ_fund = st.multiselect("Select Mutual Funds for Portfolio", FUNDS, default=FUNDS[:2], format_func=lambda x: ASSET_MAP[x])
            
        selection = univ_stock + univ_fund
        
        if st.button("Calculate Optimal Allocation"):
            if not selection: st.error("Please select at least 2 assets.")
            else:
                with st.spinner("Running Mean-Variance Optimization..."):
                    w, assets = optimize_portfolio(selection, risk_profile)
                    
                    if len(w) > 0:
                        res_df = pd.DataFrame({
                            "Asset": [ASSET_MAP.get(a,a) for a in assets],
                            "Allocation %": w,
                            "Value (₹)": w * capital
                        })
                        res_df = res_df[res_df['Allocation %'] > 0.001].sort_values("Allocation %", ascending=False)
                        
                        col_chart, col_data = st.columns([1, 1])
                        with col_chart:
                            fig_pie = px.pie(res_df, values='Allocation %', names='Asset', hole=0.4)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with col_data:
                            st.dataframe(res_df.style.format({"Allocation %": "{:.1%}", "Value (₹)": "₹{:,.2f}"}), use_container_width=True)
                        st.session_state['report_data'] = res_df
                    else:
                        st.error("Optimization failed. Try selecting different assets.")

    # --- TAB 5: REPORTS ---
    with tabs[4]:
        st.subheader("Executive Summary")
        if 'report_data' in st.session_state:
            df_rep = st.session_state['report_data']
            report_text = f"AI CAPITAL ALLOTMENT - EXECUTIVE SUMMARY\nDate: {datetime.now().strftime('%Y-%m-%d')}\nRisk Profile: {risk_profile}\nCapital: ₹{capital:,.2f}\n\nSTRATEGIC ALLOCATION:\n"
            for index, row in df_rep.iterrows():
                report_text += f"- {row['Asset']}: {row['Allocation %']*100:.1f}% (₹{row['Value (₹)']:,.2f})\n"
            st.text_area("Report Content", report_text, height=300)
            st.download_button("Export Report", report_text, file_name="Executive_Report.txt")
        else:
            st.info("Run the Allocation Engine to generate the report.")

if __name__ == "__main__":
    main()
