import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings

# ==========================================
# 0. CONFIGURATION & SUPPRESSION
# ==========================================
# Suppress warnings for clean UI (Pandas/Streamlit deprecations)
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Capital Allocation Advisor",
    layout="wide",
    page_icon="🇮🇳",
    initial_sidebar_state="expanded"
)

# Constants
RISK_FREE_RATE = 0.070  # Approx India 10Y Bond Yield

# Asset Dictionary: Maps Tickers to Real Names
# REMOVED TATAMOTORS.NS to fix 404 Error
ASSET_NAMES = {
    # Stocks (NSE)
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'TCS',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'ICICIBANK.NS': 'ICICI Bank',
    'ITC.NS': 'ITC Ltd',
    'SBIN.NS': 'SBI',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'LT.NS': 'Larsen & Toubro',
    'HINDUNILVR.NS': 'HUL',
    'AXISBANK.NS': 'Axis Bank',
    'MARUTI.NS': 'Maruti Suzuki',
    'SUNPHARMA.NS': 'Sun Pharma',
    'BAJFINANCE.NS': 'Bajaj Finance',
    
    # Mutual Funds (BSE Star MF Proxies)
    '0P0000XVAA.BO': 'HDFC Top 100 Fund',
    '0P0000XW8F.BO': 'SBI Bluechip Fund',
    '0P0000XVK5.BO': 'Axis Bluechip Fund',
    '0P0000XVTR.BO': 'ICICI Pru Value Discovery',
    '0P00005WLZ.BO': 'Nippon India Small Cap',
    '0P00009J3J.BO': 'Mirae Asset Large Cap'
}

STOCKS = [k for k in ASSET_NAMES.keys() if '.NS' in k]
MUTUAL_FUNDS = [k for k in ASSET_NAMES.keys() if '.BO' in k]

# ==========================================
# 2. DATA ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def fetch_data(tickers, period="2y"):
    """Robust data fetcher for multiple tickers."""
    if not tickers: return pd.DataFrame(), pd.DataFrame()
    try:
        # Fetch OHLC data for stocks
        data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, progress=False)
        prices = pd.DataFrame()
        
        # Helper to extract Close prices
        for t in tickers:
            try:
                if len(tickers) == 1:
                    # Single ticker structure
                    prices[t] = data['Close'] if 'Close' in data.columns else data
                else:
                    # Multi-ticker structure
                    if t in data.columns.levels[0]:
                        prices[t] = data[t]['Close']
            except: pass
        
        # Drop rows where all cols are NaN
        prices.dropna(how='all', inplace=True)
        return prices, data 
    except: return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=600)
def get_market_movers():
    """Identify Top Gainers/Losers."""
    prices, _ = fetch_data(STOCKS, period="5d")
    if prices.empty: return pd.DataFrame(), pd.DataFrame()
    
    latest = prices.iloc[-1]
    prev = prices.iloc[-2]
    
    changes = ((latest - prev) / prev) * 100
    ranking = changes.sort_values(ascending=False)
    
    # Fix FutureWarning: fill_method=None
    volatility = prices.pct_change(fill_method=None).std() * np.sqrt(252) * 100
    
    return ranking, volatility

def get_stock_fundamentals(ticker):
    """Fetch live fundamentals."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "pe": info.get('trailingPE', 0),
            "mcap": info.get('marketCap', 0),
            "high52": info.get('fiftyTwoWeekHigh', 0),
            "low52": info.get('fiftyTwoWeekLow', 0),
            "sector": info.get('sector', 'N/A')
        }
    except:
        return {}

# ==========================================
# 3. ANALYTICS & AI ENGINE
# ==========================================
def calculate_metrics(series):
    if series.empty: return 0,0,0,0
    # Fix FutureWarning: fill_method=None
    returns = series.pct_change(fill_method=None).dropna()
    ann_vol = returns.std() * np.sqrt(252)
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    sharpe = (total_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    return total_ret, ann_vol, sharpe, max_dd

def run_optimization(tickers, risk_profile):
    prices, _ = fetch_data(tickers, period="1y")
    if prices.empty: return [], []
    
    # Fix FutureWarning: fill_method=None
    returns = prices.pct_change(fill_method=None).dropna()
    mu = returns.mean() * 252
    sigma = returns.cov() * 252
    n = len(tickers)
    
    if risk_profile == "Conservative":
        bounds = tuple((0.0, 0.20) for _ in range(n))
    elif risk_profile == "Aggressive":
        bounds = tuple((0.0, 0.50) for _ in range(n))
    else: 
        bounds = tuple((0.05, 0.35) for _ in range(n))
        
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    def neg_sharpe(weights):
        p_ret = np.sum(weights * mu)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        return -(p_ret - RISK_FREE_RATE) / p_vol
    
    guess = [1./n] * n
    result = minimize(neg_sharpe, guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x, prices.columns

# ==========================================
# 4. DASHBOARD UI
# ==========================================
def main():
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.info("💡 **Instructions:** Enter your capital and risk appetite below.")
    
    capital = st.sidebar.number_input("Investment Amount (₹)", 50000, 10000000, 500000, step=50000)
    risk = st.sidebar.select_slider("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], value="Moderate")
    
    st.sidebar.markdown("### Portfolio Composition")
    inc_stock = st.sidebar.checkbox("Include Stocks", value=True)
    inc_mf = st.sidebar.checkbox("Include Mutual Funds", value=True)

    st.title("AI Capital Allocation Advisor")
    st.markdown(f"**Current Profile:** {risk} Investor | **Capital Deployed:** ₹{capital:,.0f}")
    
    tabs = st.tabs(["🔍 Explore", "📈 Stocks", "💰 Mutual Funds", "🤖 Combined Portfolio", "📝 Report"])
    
    # --- TAB 1: EXPLORE ---
    with tabs[0]:
        st.subheader("Market Intelligence Dashboard")
        c1, c2, c3 = st.columns(3)
        nifty = yf.Ticker("^NSEI").history(period="2d")
        bank = yf.Ticker("^NSEBANK").history(period="2d")
        sensex = yf.Ticker("^BSESN").history(period="2d")
        
        def get_metric(hist):
            if len(hist) < 2: return 0.0, 0.0
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            return curr, ((curr-prev)/prev)*100

        n_val, n_chg = get_metric(nifty)
        b_val, b_chg = get_metric(bank)
        s_val, s_chg = get_metric(sensex)
        
        c1.metric("NIFTY 50", f"₹{n_val:,.0f}", f"{n_chg:.2f}%")
        c2.metric("BANK NIFTY", f"₹{b_val:,.0f}", f"{b_chg:.2f}%")
        c3.metric("SENSEX", f"₹{s_val:,.0f}", f"{s_chg:.2f}%")
        
        st.divider()
        ranking, volatility = get_market_movers()
        if not ranking.empty:
            g_col, l_col, v_col = st.columns(3)
            with g_col:
                st.markdown("##### 🚀 Top Gainers (Today)")
                for t, v in ranking.head(5).items():
                    name = ASSET_NAMES.get(t, t).split('.')[0]
                    st.write(f"**{name}**: :green[+{v:.2f}%]")
            with l_col:
                st.markdown("##### 🔻 Top Losers (Today)")
                for t, v in ranking.tail(5).sort_values().items():
                    name = ASSET_NAMES.get(t, t).split('.')[0]
                    st.write(f"**{name}**: :red[{v:.2f}%]")
            with v_col:
                st.markdown("##### ⚡ Volume Shockers (High Vol)")
                for t, v in volatility.sort_values(ascending=False).head(5).items():
                    name = ASSET_NAMES.get(t, t).split('.')[0]
                    st.write(f"**{name}**: {v:.1f}% Vol")

    # --- TAB 2: STOCKS ---
    with tabs[1]:
        st.subheader("Deep Dive: Equity Analysis")
        stock_pick = st.selectbox("Select Stock to Analyze", STOCKS, format_func=lambda x: ASSET_NAMES[x])
        
        if stock_pick:
            prices_df, full_data = fetch_data([stock_pick], period="2y")
            
            if not prices_df.empty:
                # 1. Fundamentals
                with st.spinner("Fetching fundamentals..."):
                    fund_info = get_stock_fundamentals(stock_pick)
                    
                f1, f2, f3, f4 = st.columns(4)
                if fund_info:
                    f1.metric("Sector", fund_info.get('sector', 'N/A'))
                    f2.metric("P/E Ratio", f"{fund_info.get('pe', 0):.2f}")
                    f3.metric("52W High", f"₹{fund_info.get('high52', 0):,.2f}")
                    f4.metric("52W Low", f"₹{fund_info.get('low52', 0):,.2f}")
                
                st.divider()
                
                # 2. Risk Metrics
                series = prices_df[stock_pick]
                ret, vol, sharpe, dd = calculate_metrics(series)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("1Y Return", f"{ret*100:.1f}%")
                m2.metric("Annual Volatility", f"{vol*100:.1f}%")
                m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                m4.metric("Max Drawdown", f"{dd*100:.1f}%")
                
                # 3. Chart & AI
                st.subheader("Technical & AI Price Forecast")
                
                if isinstance(full_data.columns, pd.MultiIndex):
                    ohlc = full_data[stock_pick]
                else:
                    ohlc = full_data
                
                ohlc = ohlc.reset_index()
                ohlc['SMA_50'] = ohlc['Close'].rolling(window=50).mean()
                ohlc['SMA_200'] = ohlc['Close'].rolling(window=200).mean()
                
                # AI Linear Regression
                ohlc['ID'] = np.arange(len(ohlc))
                clean_df = ohlc.dropna(subset=['Close'])
                X = clean_df[['ID']]
                y = clean_df['Close']
                model = LinearRegression().fit(X, y)
                
                last_id = ohlc['ID'].iloc[-1]
                future_ids = np.arange(last_id, last_id + 30).reshape(-1, 1)
                future_prices = model.predict(future_ids)
                future_dates = [ohlc['Date'].iloc[-1] + timedelta(days=i) for i in range(1, 31)]
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=ohlc['Date'], open=ohlc['Open'], high=ohlc['High'],
                    low=ohlc['Low'], close=ohlc['Close'], name='Price'
                ))
                fig.add_trace(go.Scatter(x=ohlc['Date'], y=ohlc['SMA_50'], line=dict(color='orange', width=1.5), name='SMA 50'))
                fig.add_trace(go.Scatter(x=ohlc['Date'], y=ohlc['SMA_200'], line=dict(color='blue', width=1.5), name='SMA 200'))
                
                trend_hist = model.predict(X)
                fig.add_trace(go.Scatter(x=clean_df['Date'], y=trend_hist, line=dict(color='green', dash='dot'), name='AI Trend (Hist)'))
                fig.add_trace(go.Scatter(x=future_dates, y=future_prices, line=dict(color='green', width=2), name='AI Forecast (30D)'))
                
                fig.update_layout(title=f"{stock_pick} - Technicals + AI Trend", xaxis_rangeslider_visible=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""ℹ️ **Chart Guide:** The **Candlesticks** show daily price action. 
                The **Orange/Blue lines** are Moving Averages (50/200 days). 
                The **Green Dotted Line** is the AI Linear Regression trend projecting 30 days ahead.""")

    # --- TAB 3: MUTUAL FUNDS ---
    with tabs[2]:
        st.subheader("Mutual Fund Analyzer")
        st.caption("Showing 1-Year Performance for Top Indian Mutual Funds")
        prices_mf, _ = fetch_data(MUTUAL_FUNDS, period="1y")
        
        if not prices_mf.empty:
            stats = []
            for mf in MUTUAL_FUNDS:
                if mf in prices_mf:
                    ret, vol, sharpe, _ = calculate_metrics(prices_mf[mf])
                    stats.append({
                        "Fund Name": ASSET_NAMES[mf],
                        "1Y Return": f"{ret*100:.1f}%",
                        "Risk": f"{vol*100:.1f}%",
                        "Sharpe": f"{sharpe:.2f}"
                    })
            st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)
            norm = prices_mf / prices_mf.iloc[0] * 100
            norm.columns = [ASSET_NAMES.get(c, c) for c in norm.columns]
            st.plotly_chart(px.line(norm, title="Normalized Growth (Base=100)"), use_container_width=True)

    # --- TAB 4: COMBINED PORTFOLIO ---
    with tabs[3]:
        st.subheader("🤖 AI Combined Portfolio Engine")
        
        universe = []
        if inc_stock: universe += STOCKS[:10]
        if inc_mf: universe += MUTUAL_FUNDS
        
        if not universe:
            st.warning("⚠️ Select at least one asset class in the sidebar.")
        else:
            if st.button("Generate Optimal Allocation"):
                with st.spinner(f"Optimizing for {risk} Profile..."):
                    weights, assets = run_optimization(universe, risk)
                    
                    allocation = pd.DataFrame({
                        "Asset": [ASSET_NAMES.get(a, a) for a in assets],
                        "Weight": weights,
                        "Value": weights * capital
                    })
                    allocation = allocation[allocation['Weight'] > 0.01].sort_values("Weight", ascending=False)
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        fig = px.pie(allocation, values='Weight', names='Asset', hole=0.4, title="Recommended Allocation")
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        st.dataframe(allocation.style.format({"Weight": "{:.1%}", "Value": "₹{:.2f}"}), use_container_width=True)
                        st.success(f"**AI Insight:** Top holding: {allocation.iloc[0]['Asset']}. Optimized for Sharpe Ratio.")

    # --- TAB 5: REPORT ---
    with tabs[4]:
        st.subheader("📄 Investment Memorandum")
        if 'allocation' in locals() and not allocation.empty:
            report = f"""
            ### AI CAPITAL ALLOCATION REPORT
            **Date:** {datetime.now().strftime('%Y-%m-%d')}
            **Client Risk Profile:** {risk}
            **Total Investment:** ₹{capital:,.0f}
            
            **Top 3 Holdings:**
            1. {allocation.iloc[0]['Asset']} ({allocation.iloc[0]['Weight']*100:.1f}%)
            2. {allocation.iloc[1]['Asset']} ({allocation.iloc[1]['Weight']*100:.1f}%)
            3. {allocation.iloc[2]['Asset']} ({allocation.iloc[2]['Weight']*100:.1f}%)
            """
            st.text_area("Copy Report", report, height=300)
            st.download_button("Download Report", report, file_name="Allocation_Report.txt")
        else:
            st.write("Generate portfolio in previous tab to view report.")

if __name__ == "__main__":
    main()
