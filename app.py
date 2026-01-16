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
import time

# ==========================================
# 0. CONFIGURATION & AUTHENTICATION
# ==========================================
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title=" AI Capital & Trade Advisor",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# --- Auth System ---
USER_DB_FILE = "users_db.json"

def load_users():
    if not os.path.exists(USER_DB_FILE): return {}
    with open(USER_DB_FILE, "r") as f: return json.load(f)

def save_users(users):
    with open(USER_DB_FILE, "w") as f: json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_page():
    st.title("🔐 Login to Friday AI")
    t1, t2 = st.tabs(["Login", "Sign Up"])
    with t1:
        u = st.text_input("Username", key="l_u")
        p = st.text_input("Password", type="password", key="l_p")
        if st.button("Login"):
            db = load_users()
            if u in db and db[u] == hash_password(p):
                st.session_state["auth"] = True
                st.session_state["user"] = u
                st.rerun()
            else: st.error("Invalid credentials")
    with t2:
        nu = st.text_input("New Username", key="s_u")
        np_pass = st.text_input("New Password", type="password", key="s_p")
        if st.button("Create Account"):
            db = load_users()
            if nu in db: st.error("User exists")
            else:
                db[nu] = hash_password(np_pass)
                save_users(db)
                st.success("Account created! Log in now.")

if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    login_page()
    st.stop()

# ==========================================
# 1. CONSTANTS & ASSETS
# ==========================================
RISK_FREE_RATE = 0.070 

ASSET_NAMES = {
    # Stocks
    'RELIANCE.NS': 'Reliance Industries', 'TCS.NS': 'TCS', 'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys', 'ICICIBANK.NS': 'ICICI Bank', 'ITC.NS': 'ITC Ltd',
    'SBIN.NS': 'SBI', 'BHARTIARTL.NS': 'Bharti Airtel', 'LT.NS': 'Larsen & Toubro',
    'HINDUNILVR.NS': 'HUL', 'AXISBANK.NS': 'Axis Bank', 'MARUTI.NS': 'Maruti Suzuki',
    'SUNPHARMA.NS': 'Sun Pharma', 'BAJFINANCE.NS': 'Bajaj Finance', 'TATAMOTORS.NS': 'Tata Motors',
    
    # Mutual Funds
    '0P0000XVAA.BO': 'HDFC Top 100 Fund', '0P0000XW8F.BO': 'SBI Bluechip Fund',
    '0P0000XVK5.BO': 'Axis Bluechip Fund', '0P0000XVTR.BO': 'ICICI Pru Value Discovery',
    '0P00005WLZ.BO': 'Nippon India Small Cap', '0P00009J3J.BO': 'Mirae Asset Large Cap'
}

STOCKS = [k for k in ASSET_NAMES.keys() if '.NS' in k]
MUTUAL_FUNDS = [k for k in ASSET_NAMES.keys() if '.BO' in k]

# ==========================================
# 2. DATA ENGINE
# ==========================================
@st.cache_data(ttl=60)
def fetch_live_price(ticker):
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if not data.empty:
            return data['Close'].iloc[-1].item() if hasattr(data['Close'].iloc[-1], 'item') else data['Close'].iloc[-1]
        return 0.0
    except: return 0.0

@st.cache_data(ttl=3600)
def fetch_bulk_data(tickers, period="2y"):
    """Optimized for Portfolio Engine"""
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, progress=False)
        prices = pd.DataFrame()
        for t in tickers:
            try:
                if len(tickers) == 1:
                    col = data['Close'] if 'Close' in data.columns else data
                else:
                    col = data[t]['Close'] if t in data.columns.levels[0] else None
                if col is not None: prices[t] = col
            except: pass
        prices.dropna(how='all', inplace=True)
        return prices
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_single_stock_data(ticker, period="1y"):
    """Optimized for Technical Analysis"""
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            try: data = data.xs(ticker, axis=1, level=0)
            except: data.columns = data.columns.get_level_values(0)
        return data
    except: return pd.DataFrame()

def get_market_status():
    hour = datetime.now().hour
    return "🟢 Market Open" if 9 <= hour < 16 else "🔴 Market Closed"

# ==========================================
# 3. ANALYTICS ENGINE
# ==========================================
def calculate_metrics(series):
    if series.empty: return 0,0,0,0
    returns = series.pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(252)
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    sharpe = (total_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    cum = (1 + returns).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    return total_ret, ann_vol, sharpe, max_dd

def run_optimization(tickers, risk_profile):
    prices = fetch_bulk_data(tickers, period="1y")
    if prices.empty or len(prices.columns) < 2: return [], []
    
    returns = prices.pct_change().dropna()
    mu, sigma = returns.mean() * 252, returns.cov() * 252
    n = len(prices.columns)
    
    bounds = ((0.0, 0.20),) * n if risk_profile == "Conservative" else \
             ((0.0, 0.50),) * n if risk_profile == "Aggressive" else ((0.05, 0.35),) * n
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    def neg_sharpe(w):
        return -((np.sum(w * mu) - RISK_FREE_RATE) / np.sqrt(np.dot(w.T, np.dot(sigma, w))))
    
    try:
        res = minimize(neg_sharpe, [1./n]*n, bounds=bounds, constraints=constraints, method='SLSQP')
        return res.x, prices.columns
    except: return [], []

# ==========================================
# 4. UI & DASHBOARD
# ==========================================
def main():
    # --- Sidebar ---
    st.sidebar.title(f"👤 {st.session_state['user']}")
    if st.sidebar.button("Logout"):
        st.session_state["auth"] = False
        st.rerun()
    st.sidebar.divider()
    
    st.sidebar.header("💰 Portfolio Settings")
    capital = st.sidebar.number_input("Capital (₹)", 50000, 10000000, 500000, step=50000)
    risk = st.sidebar.select_slider("Risk Profile", ["Conservative", "Moderate", "Aggressive"], value="Moderate")
    
    st.title("Friday: AI Financial Core")
    st.markdown(f"**Status:** {get_market_status()} | **Capital:** ₹{capital:,.0f} | **Risk:** {risk}")
    
    tabs = st.tabs(["📊 Dashboard", "📉 Pro Stock Analysis", "🏢 Mutual Funds", "🤖 AI Allocator", "📝 Report"])
    
    # --- TAB 1: DASHBOARD ---
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        try:
            nifty = yf.Ticker("^NSEI").history(period="2d")['Close']
            n_val, n_chg = nifty.iloc[-1], ((nifty.iloc[-1]-nifty.iloc[-2])/nifty.iloc[-2])*100
            c1.metric("NIFTY 50", f"₹{n_val:,.0f}", f"{n_chg:.2f}%")
        except: c1.metric("NIFTY 50", "Error", "0%")
        
        # Market Movers Logic
        st.subheader("Market Movers")
        movers_df = fetch_bulk_data(STOCKS, "5d")
        if not movers_df.empty:
            chg = ((movers_df.iloc[-1] - movers_df.iloc[-2]) / movers_df.iloc[-2]) * 100
            top = chg.sort_values(ascending=False).head(5)
            bot = chg.sort_values().head(5)
            
            mc1, mc2 = st.columns(2)
            with mc1:
                st.write("🚀 **Top Gainers**")
                for t, v in top.items(): st.write(f"{ASSET_NAMES.get(t,t)}: :green[+{v:.2f}%]")
            with mc2:
                st.write("🔻 **Top Losers**")
                for t, v in bot.items(): st.write(f"{ASSET_NAMES.get(t,t)}: :red[{v:.2f}%]")

    # --- TAB 2: PRO STOCK ANALYSIS (Merged Feature) ---
    with tabs[1]:
        col_sel, col_live = st.columns([2, 1])
        with col_sel:
            stock_pick = st.selectbox("Select Stock", STOCKS, format_func=lambda x: ASSET_NAMES[x])
        with col_live:
            if st.button("🔄 Refresh Live Price"): st.rerun()
            
        data = fetch_single_stock_data(stock_pick)
        if not data.empty:
            # LIVE PRICE HEADER
            lp = fetch_live_price(stock_pick)
            prev_close = data['Close'].iloc[-1] # Using last hist close as ref if live fails or is same
            diff = lp - prev_close
            st.metric(f"Live Price: {ASSET_NAMES[stock_pick]}", f"₹{lp:,.2f}", f"{(diff/prev_close)*100:.2f}%")

            # TECHNICALS
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            bb = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
            data['BB_H'] = bb.bollinger_hband()
            data['BB_L'] = bb.bollinger_lband()

            # 1. MAIN CHART (Candles + BB + SMA)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                         low=data['Low'], close=data['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_H'], line=dict(color='gray', width=1, dash='dot'), name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_L'], line=dict(color='gray', width=1, dash='dot'), name='BB Lower'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='orange', width=2), name='SMA 50'))
            
            # AI PREDICTION OVERLAY
            data['ID'] = np.arange(len(data))
            model = LinearRegression().fit(data[['ID']], data['Close'])
            fut_id = np.arange(data['ID'].iloc[-1], data['ID'].iloc[-1]+30).reshape(-1,1)
            fut_dates = [data.index[-1] + timedelta(days=i) for i in range(1,31)]
            fig.add_trace(go.Scatter(x=fut_dates, y=model.predict(fut_id), line=dict(color='cyan', width=2, dash='dash'), name='AI Forecast (30D)'))
            
            fig.update_layout(title="Price Action + AI Trend", height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. SUB-CHARTS (RSI & Patterns)
            c_tech1, c_tech2 = st.columns(2)
            with c_tech1:
                # RSI
                fig_rsi = go.Figure(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_color='red', line_dash='dot')
                fig_rsi.add_hline(y=30, line_color='green', line_dash='dot')
                fig_rsi.update_layout(title="RSI Momentum", height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with c_tech2:
                # Pattern Recognition
                order = 5
                min_idx = argrelextrema(data['Close'].values, np.less, order=order)[0]
                max_idx = argrelextrema(data['Close'].values, np.greater, order=order)[0]
                
                fig_pat = go.Figure()
                fig_pat.add_trace(go.Scatter(x=data.index, y=data['Close'], line=dict(color='gray', width=1)))
                fig_pat.add_trace(go.Scatter(x=data.index[min_idx], y=data['Close'].iloc[min_idx], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Support'))
                fig_pat.add_trace(go.Scatter(x=data.index[max_idx], y=data['Close'].iloc[max_idx], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Resistance'))
                fig_pat.update_layout(title="Support & Resistance Detection", height=300)
                st.plotly_chart(fig_pat, use_container_width=True)
            
            # FUNDAMENTALS
            with st.expander("📊 Fundamental Data"):
                t = yf.Ticker(stock_pick)
                info = t.info
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
                fc2.metric("Market Cap", f"₹{info.get('marketCap', 0)/1e7:,.0f} Cr")
                fc3.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%")

    # --- TAB 3: MUTUAL FUNDS ---
    with tabs[2]:
        mf_data = fetch_bulk_data(MUTUAL_FUNDS, "1y")
        if not mf_data.empty:
            norm = mf_data / mf_data.iloc[0] * 100
            norm.columns = [ASSET_NAMES.get(c,c) for c in norm.columns]
            st.plotly_chart(px.line(norm, title="Mutual Fund Performance (Rebased to 100)"), use_container_width=True)
            
            stats = []
            for c in mf_data.columns:
                r, v, s, _ = calculate_metrics(mf_data[c])
                stats.append({"Fund": ASSET_NAMES[c], "Return": f"{r:.1%}", "Risk": f"{v:.1%}", "Sharpe": f"{s:.2f}"})
            st.dataframe(pd.DataFrame(stats))

    # --- TAB 4: AI ALLOCATOR ---
    with tabs[3]:
        st.subheader("🤖 Portfolio Optimization Engine")
        universe = []
        if st.checkbox("Include Stocks", True): universe += STOCKS[:10] # Limit to 10 for speed
        if st.checkbox("Include Mutual Funds", True): universe += MUTUAL_FUNDS
        
        if st.button("Run AI Optimization"):
            with st.spinner("Analyzing correlation matrices..."):
                weights, assets = run_optimization(universe, risk)
                if len(weights) > 0:
                    alloc_df = pd.DataFrame({"Asset": [ASSET_NAMES.get(a,a) for a in assets], "Weight": weights, "Value": weights * capital})
                    alloc_df = alloc_df[alloc_df.Weight > 0.01].sort_values("Weight", ascending=False)
                    
                    ac1, ac2 = st.columns([1,1])
                    ac1.plotly_chart(px.pie(alloc_df, values='Weight', names='Asset', title="Optimal Allocation"), use_container_width=True)
                    ac2.dataframe(alloc_df.style.format({"Weight": "{:.1%}", "Value": "₹{:.0f}"}), use_container_width=True)
                    
                    # Store for report
                    st.session_state['last_alloc'] = alloc_df

    # --- TAB 5: REPORT ---
    with tabs[4]:
        if 'last_alloc' in st.session_state:
            df = st.session_state['last_alloc']
            report = f"""
            AI INVESTMENT MEMORANDUM
            ------------------------
            Date: {datetime.now().strftime('%Y-%m-%d')}
            Investor: {st.session_state['user']}
            Profile: {risk} | Capital: ₹{capital:,.0f}
            
            STRATEGY:
            Optimized for maximum Sharpe Ratio (Return/Risk).
            
            ALLOCATION:
            {df.to_string(index=False)}
            """
            st.text_area("Report Preview", report, height=300)
            st.download_button("Download PDF/TXT", report, "Friday_Report.txt")
        else:
            st.info("Please run the AI Allocator in Tab 4 first.")

if __name__ == "__main__":
    main()
