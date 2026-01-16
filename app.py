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
    page_title="Strategic Capital Allocation System",
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
    st.header("System Access")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Access Dashboard"):
            db = load_users()
            if u in db and db[u] == hash_password(p):
                st.session_state["authenticated"] = True
                st.session_state["username"] = u
                st.rerun()
            else:
                st.error("Authentication Failed: Invalid credentials.")
    
    with tab2:
        nu = st.text_input("New Username", key="reg_user")
        np_pass = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Create Account"):
            db = load_users()
            if nu in db:
                st.error("Error: User already exists.")
            else:
                db[nu] = hash_password(np_pass)
                save_users(db)
                st.success("Account created successfully. Please login.")

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
    # Stocks
    'RELIANCE.NS': 'Reliance Industries', 'TCS.NS': 'TCS', 'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys', 'ICICIBANK.NS': 'ICICI Bank', 'ITC.NS': 'ITC Ltd',
    'SBIN.NS': 'SBI', 'BHARTIARTL.NS': 'Bharti Airtel', 'LT.NS': 'Larsen & Toubro',
    'HINDUNILVR.NS': 'HUL', 'AXISBANK.NS': 'Axis Bank', 'MARUTI.NS': 'Maruti Suzuki',
    'SUNPHARMA.NS': 'Sun Pharma', 'BAJFINANCE.NS': 'Bajaj Finance', 'TATAMOTORS.NS': 'Tata Motors',
    
    # Mutual Funds
    '0P0000XVAA.BO': 'HDFC Top 100', '0P0000XW8F.BO': 'SBI Bluechip',
    '0P0000XVK5.BO': 'Axis Bluechip', '0P0000XVTR.BO': 'ICICI Pru Value',
    '0P00005WLZ.BO': 'Nippon Small Cap', '0P00009J3J.BO': 'Mirae Large Cap'
}

STOCKS = [k for k in ASSET_MAP.keys() if '.NS' in k]
FUNDS = [k for k in ASSET_MAP.keys() if '.BO' in k]

# ==========================================
# 2. DATA ENGINE (ROBUST)
# ==========================================
def fetch_index_data(ticker):
    """
    Dedicated function for Indices (Nifty/Sensex) which often fail on standard calls.
    Fetches 5 days of history to ensure we find the Last Traded Price (LTP).
    """
    try:
        idx = yf.Ticker(ticker)
        # Fetch 5 days to cover weekends/holidays
        hist = idx.history(period="5d")
        if hist.empty:
            return 0.0, 0.0
        
        curr = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2] if len(hist) > 1 else curr
        change_pct = ((curr - prev) / prev) * 100
        return curr, change_pct
    except:
        return 0.0, 0.0

@st.cache_data(ttl=60)
def fetch_live_price(ticker):
    """Fetches latest available price for stocks."""
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if not data.empty:
            price = data['Close'].iloc[-1]
            # Handle single value extraction safely
            return float(price.item()) if hasattr(price, 'item') else float(price)
        # Fallback to daily data if intraday fails
        data_d = yf.download(ticker, period="1d", progress=False)
        return float(data_d['Close'].iloc[-1])
    except:
        return 0.0

@st.cache_data(ttl=3600)
def fetch_historical_matrix(tickers, period="2y"):
    """Fetches clean Close price matrix for optimization."""
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, progress=False)
        prices = pd.DataFrame()
        for t in tickers:
            try:
                # Handle MultiIndex logic
                if len(tickers) == 1:
                    series = data['Close'] if 'Close' in data.columns else data
                else:
                    series = data[t]['Close'] if t in data.columns.levels[0] else None
                
                if series is not None:
                    prices[t] = series
            except: pass
        
        # Forward fill to handle holidays, then drop remaining NaNs
        prices.ffill(inplace=True)
        prices.dropna(how='any', inplace=True)
        return prices
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_analysis_data(ticker):
    """Deep fetch for technical analysis."""
    try:
        data = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        # Flatten MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex):
             # Try to get level 0 if it matches ticker, else just drop levels
            try: data = data.xs(ticker, axis=1, level=0)
            except: data.columns = data.columns.get_level_values(0)
        return data
    except: return pd.DataFrame()

# ==========================================
# 3. ANALYTICS CORE
# ==========================================
def calculate_metrics(series):
    if series.empty: return 0,0,0,0
    returns = series.pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(252)
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    sharpe = (total_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    return total_ret, ann_vol, sharpe, 0

def optimize_portfolio(tickers, risk_level):
    df = fetch_historical_matrix(tickers, "1y")
    if df.empty or len(df.columns) < 2: return [], []
    
    returns = df.pct_change().dropna()
    mu = returns.mean() * 252
    sigma = returns.cov() * 252
    n_assets = len(df.columns)
    
    # Allocations Constraints
    if risk_level == "Conservative":
        bounds = tuple((0.0, 0.15) for _ in range(n_assets))
    elif risk_level == "Aggressive":
        bounds = tuple((0.0, 0.40) for _ in range(n_assets))
    else: # Moderate
        bounds = tuple((0.0, 0.25) for _ in range(n_assets))
        
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    def neg_sharpe_ratio(weights):
        p_ret = np.sum(weights * mu)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        return -((p_ret - RISK_FREE_RATE) / p_vol)
    
    try:
        init_guess = [1./n_assets for _ in range(n_assets)]
        result = minimize(neg_sharpe_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x, df.columns
    except: return [], []

# ==========================================
# 4. DASHBOARD UI
# ==========================================
def main():
    # --- Sidebar ---
    st.sidebar.header("Parameters")
    st.sidebar.markdown(f"User: **{st.session_state['username']}**")
    
    capital = st.sidebar.number_input("Capital Allocation (₹)", 50000, 100000000, 500000, step=100000)
    risk_profile = st.sidebar.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
    
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

    # --- Main Layout ---
    st.title("Strategic Capital Allocation System")
    st.markdown("---")

    tabs = st.tabs(["Market Overview", "Stock Analysis", "Mutual Funds", "Portfolio Optimizer", "Report Generation"])
    
    # --- TAB 1: MARKET OVERVIEW ---
    with tabs[0]:
        st.subheader("Market Indices")
        c1, c2, c3 = st.columns(3)
        
        # Robust fetch for indices
        n_val, n_pct = fetch_index_data("^NSEI")
        b_val, b_pct = fetch_index_data("^NSEBANK")
        s_val, s_pct = fetch_index_data("^BSESN")
        
        c1.metric("NIFTY 50", f"₹{n_val:,.0f}", f"{n_pct:.2f}%")
        c2.metric("BANK NIFTY", f"₹{b_val:,.0f}", f"{b_pct:.2f}%")
        c3.metric("SENSEX", f"₹{s_val:,.0f}", f"{s_pct:.2f}%")
        
        st.markdown("#### Market Momentum (Top 5 Nifty Stocks)")
        movers_data = fetch_historical_matrix(STOCKS, "5d")
        if not movers_data.empty:
            # Calculate % change from open of 5 days ago to now
            changes = ((movers_data.iloc[-1] - movers_data.iloc[0]) / movers_data.iloc[0]) * 100
            top_g = changes.sort_values(ascending=False).head(5)
            
            # Simple Table
            st.table(pd.DataFrame({
                "Asset": [ASSET_MAP.get(t,t) for t in top_g.index],
                "Ticker": top_g.index,
                "5-Day Change (%)": [f"{x:.2f}%" for x in top_g.values]
            }))

    # --- TAB 2: STOCK ANALYSIS ---
    with tabs[1]:
        col_ctrl, col_disp = st.columns([1, 3])
        with col_ctrl:
            selected_stock = st.selectbox("Select Asset", STOCKS, format_func=lambda x: ASSET_MAP[x])
            if st.button("Refresh Data"): st.cache_data.clear()
            
        with col_disp:
            df_stock = fetch_analysis_data(selected_stock)
            if not df_stock.empty:
                curr_price = fetch_live_price(selected_stock)
                
                st.metric(ASSET_MAP[selected_stock], f"₹{curr_price:,.2f}")
                
                # Tech Indicators
                df_stock['SMA_50'] = ta.trend.sma_indicator(df_stock['Close'], window=50)
                df_stock['RSI'] = ta.momentum.rsi(df_stock['Close'], window=14)
                
                # Charting
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_stock.index,
                                open=df_stock['Open'], high=df_stock['High'],
                                low=df_stock['Low'], close=df_stock['Close'],
                                name="OHLC"))
                fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['SMA_50'], 
                                         line=dict(color='orange', width=1.5), name="SMA 50"))
                
                # Linear Regression Projection
                df_stock['idx'] = np.arange(len(df_stock))
                clean = df_stock.dropna(subset=['Close'])
                reg = LinearRegression().fit(clean[['idx']], clean['Close'])
                
                # Forecast next 30 days
                last_idx = clean['idx'].iloc[-1]
                future_idx = np.arange(last_idx, last_idx + 30).reshape(-1, 1)
                forecast = reg.predict(future_idx)
                future_dates = [clean.index[-1] + timedelta(days=i) for i in range(1, 31)]
                
                fig.add_trace(go.Scatter(x=future_dates, y=forecast, 
                                         line=dict(color='blue', dash='dot'), name="Linear Forecast"))

                fig.update_layout(height=500, xaxis_rangeslider_visible=False, 
                                  margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Fundamentals
                try:
                    info = yf.Ticker(selected_stock).info
                    f1, f2, f3, f4 = st.columns(4)
                    f1.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
                    f2.metric("Market Cap", f"₹{info.get('marketCap', 0)/1e9:.1f}B")
                    f3.metric("52W High", f"₹{info.get('fiftyTwoWeekHigh', 0):,.0f}")
                    f4.metric("Sector", info.get('sector', 'N/A'))
                except:
                    st.warning("Fundamental data unavailable.")

    # --- TAB 3: MUTUAL FUNDS ---
    with tabs[2]:
        mf_df = fetch_historical_matrix(FUNDS, "1y")
        if not mf_df.empty:
            st.subheader("Fund Performance (Normalized)")
            norm_mf = mf_df / mf_df.iloc[0] * 100
            norm_mf.columns = [ASSET_MAP.get(c,c) for c in norm_mf.columns]
            st.line_chart(norm_mf)

    # --- TAB 4: OPTIMIZER ---
    with tabs[3]:
        st.subheader("Mean-Variance Optimization")
        
        univ_stock = st.multiselect("Select Stocks", STOCKS, default=STOCKS[:5], format_func=lambda x: ASSET_MAP[x])
        univ_fund = st.multiselect("Select Funds", FUNDS, default=FUNDS[:2], format_func=lambda x: ASSET_MAP[x])
        
        selection = univ_stock + univ_fund
        
        if st.button("Execute Optimization"):
            if not selection:
                st.error("Please select at least 2 assets.")
            else:
                with st.spinner("Calculating covariance matrix..."):
                    w, assets = optimize_portfolio(selection, risk_profile)
                    
                    if len(w) > 0:
                        res_df = pd.DataFrame({
                            "Asset": [ASSET_MAP.get(a,a) for a in assets],
                            "Allocation %": w,
                            "Value (₹)": w * capital
                        })
                        res_df = res_df[res_df['Allocation %'] > 0.001].sort_values("Allocation %", ascending=False)
                        
                        c_pie, c_table = st.columns([1, 1])
                        
                        with c_pie:
                            fig_pie = px.pie(res_df, values='Allocation %', names='Asset', 
                                             title="Recommended Allocation")
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with c_table:
                            st.dataframe(res_df.style.format({"Allocation %": "{:.1%}", "Value (₹)": "₹{:,.2f}"}), 
                                         use_container_width=True)
                            
                        st.session_state['report_data'] = res_df
                    else:
                        st.error("Optimization failed. Insufficient historical data for selected assets.")

    # --- TAB 5: REPORT ---
    with tabs[4]:
        st.subheader("Investment Memorandum")
        if 'report_data' in st.session_state:
            df_rep = st.session_state['report_data']
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            report_text = f"""
            STRATEGIC CAPITAL ALLOCATION REPORT
            -----------------------------------
            Date: {now_str}
            Profile: {risk_profile}
            Total Capital: ₹{capital:,.2f}
            
            ALLOCATION STRATEGY:
            """
            for index, row in df_rep.iterrows():
                report_text += f"\n- {row['Asset']}: {row['Allocation %']*100:.1f}%  (₹{row['Value (₹)']:,.2f})"
            
            st.text_area("Report Content", report_text, height=300)
            st.download_button("Export Report", report_text, file_name=f"Allocation_Report_{datetime.now().date()}.txt")
        else:
            st.info("No optimization data found. Please run the Optimizer first.")

if __name__ == "__main__":
    main()
