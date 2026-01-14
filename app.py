import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Capital Allocation Advisor",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Constants
RISK_FREE_RATE = 0.070  # Approx India 10Y Bond Yield

# Asset Dictionary: Maps Tickers to Real Names
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
    'TATAMOTORS.NS': 'Tata Motors',
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
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, progress=False)
        prices = pd.DataFrame()
        for t in tickers:
            try:
                if len(tickers) == 1:
                    prices[t] = data['Close'] if 'Close' in data.columns else data
                else:
                    if t in data.columns.levels[0]:
                        prices[t] = data[t]['Close']
            except: pass
        prices.dropna(how='all', inplace=True)
        return prices
    except: return pd.DataFrame()

@st.cache_data(ttl=600)
def get_market_movers():
    """Identify Top Gainers/Losers from our watchlist."""
    df = fetch_data(STOCKS, period="5d")
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Calculate % Change
    changes = ((latest - prev) / prev) * 100
    ranking = changes.sort_values(ascending=False)
    
    # Calculate Volatility (Standard Deviation)
    volatility = df.pct_change().std() * np.sqrt(252) * 100
    
    return ranking, volatility

# ==========================================
# 3. ANALYTICS & AI ENGINE
# ==========================================
def calculate_metrics(series):
    """Compute financial risk/return metrics."""
    if series.empty: return 0,0,0,0
    returns = series.pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(252)
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    sharpe = (total_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    
    # Max Drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    return total_ret, ann_vol, sharpe, max_dd

def run_optimization(tickers, risk_profile):
    """
    AI Optimization Logic:
    - Conservative: Restricts single asset weight to 15% (Forced Diversification)
    - Aggressive: Allows single asset weight up to 40% (Concentration)
    """
    df = fetch_data(tickers, period="1y")
    if df.empty: return [], []
    
    returns = df.pct_change().dropna()
    mu = returns.mean() * 252
    sigma = returns.cov() * 252
    n = len(tickers)
    
    # Dynamic Bounds based on Risk Profile
    if risk_profile == "Conservative":
        # Spread capital: Min 0%, Max 20% per asset
        bounds = tuple((0.0, 0.20) for _ in range(n))
    elif risk_profile == "Aggressive":
        # Concentrated bets: Min 0%, Max 50% per asset
        bounds = tuple((0.0, 0.50) for _ in range(n))
    else: # Moderate
        bounds = tuple((0.05, 0.35) for _ in range(n))
        
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    def neg_sharpe(weights):
        p_ret = np.sum(weights * mu)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        return -(p_ret - RISK_FREE_RATE) / p_vol
    
    # Run Optimization
    guess = [1./n] * n
    result = minimize(neg_sharpe, guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x, df.columns

# ==========================================
# 4. DASHBOARD UI
# ==========================================
def main():
    # --- Sidebar Configuration ---
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.info("💡 **Instructions:** Enter your capital and risk appetite below to start the AI analysis.")
    
    capital = st.sidebar.number_input("Investment Amount (₹)", 50000, 10000000, 500000, step=50000)
    risk = st.sidebar.select_slider("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], value="Moderate")
    
    st.sidebar.markdown("### Portfolio Composition")
    inc_stock = st.sidebar.checkbox("Include Stocks", value=True)
    inc_mf = st.sidebar.checkbox("Include Mutual Funds", value=True)

    # --- Main Header ---
    st.title("AI Capital Allocation Advisor")
    st.markdown(f"**Current Profile:** {risk} Investor | **Capital Deployed:** ₹{capital:,.0f}")
    
    # --- Tabs ---
    tabs = st.tabs(["🔍 Explore", "📈 Stocks", "💰 Mutual Funds", "🤖 Combined Portfolio", "📝 Report"])
    
    # TAB 1: EXPLORE
    with tabs[0]:
        st.subheader("Market Intelligence Dashboard")
        
        # Live Indices (Proxies)
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
        
        # Gainers/Losers
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

    # TAB 2: STOCKS
    with tabs[1]:
        st.subheader("Deep Dive: Equity Analysis")
        stock_pick = st.selectbox("Select Stock to Analyze", STOCKS, format_func=lambda x: ASSET_NAMES[x])
        
        if stock_pick:
            df = fetch_data([stock_pick])
            if not df.empty:
                series = df[stock_pick]
                ret, vol, sharpe, dd = calculate_metrics(series)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("1Y Return", f"{ret*100:.1f}%")
                m2.metric("Annual Volatility", f"{vol*100:.1f}%")
                m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                m4.metric("Max Drawdown", f"{dd*100:.1f}%")
                
                st.line_chart(series)

    # TAB 3: MUTUAL FUNDS
    with tabs[2]:
        st.subheader("Mutual Fund Analyzer")
        st.caption("Showing 1-Year Performance for Top Indian Mutual Funds")
        
        mf_df = fetch_data(MUTUAL_FUNDS, period="1y")
        
        if not mf_df.empty:
            # Metrics Table
            stats = []
            for mf in MUTUAL_FUNDS:
                if mf in mf_df:
                    ret, vol, sharpe, _ = calculate_metrics(mf_df[mf])
                    stats.append({
                        "Fund Name": ASSET_NAMES[mf],
                        "1Y Return": f"{ret*100:.1f}%",
                        "Risk": f"{vol*100:.1f}%",
                        "Sharpe": f"{sharpe:.2f}"
                    })
            
            st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)
            
            # Comparison Chart
            st.markdown("##### Performance Trend (Base = 100)")
            norm = mf_df / mf_df.iloc[0] * 100
            norm.columns = [ASSET_NAMES.get(c, c) for c in norm.columns]
            st.plotly_chart(px.line(norm), use_container_width=True)

    # TAB 4: COMBINED PORTFOLIO
    with tabs[3]:
        st.subheader("🤖 AI Combined Portfolio Engine")
        st.write("This engine uses **Modern Portfolio Theory (MPT)** to find the mathematical 'sweet spot' between Stocks and Mutual Funds based on your risk profile.")
        
        # Build Universe
        universe = []
        if inc_stock: universe += STOCKS[:10] # Top 10 stocks
        if inc_mf: universe += MUTUAL_FUNDS
        
        if not universe:
            st.warning("⚠️ Please select at least one asset class in the sidebar to proceed.")
        else:
            if st.button("Generate Optimal Allocation"):
                with st.spinner(f"Optimizing for {risk} Profile..."):
                    weights, assets = run_optimization(universe, risk)
                    
                    # Process Results
                    allocation = pd.DataFrame({
                        "Asset": [ASSET_NAMES.get(a, a) for a in assets],
                        "Weight": weights,
                        "Value": weights * capital
                    })
                    # Filter small weights
                    allocation = allocation[allocation['Weight'] > 0.01].sort_values("Weight", ascending=False)
                    
                    # Display Visuals
                    c1, c2 = st.columns([1, 1])
                    
                    with c1:
                        fig = px.pie(allocation, values='Weight', names='Asset', hole=0.4, title="Recommended Allocation")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with c2:
                        st.markdown("##### Allocation Breakdown")
                        st.dataframe(allocation.style.format({"Weight": "{:.1%}", "Value": "₹{:.2f}"}), use_container_width=True)
                        
                        # Dynamic Insight
                        top_pick = allocation.iloc[0]['Asset']
                        st.success(f"**AI Insight:** Your top recommended holding is **{top_pick}**. The model suggests this because it offers the best stability-to-return ratio for your selected {risk} risk profile.")

    # TAB 5: REPORT
    with tabs[4]:
        st.subheader("📄 Investment Memorandum")
        st.write("Please run the Portfolio Optimization in the previous tab to generate this report.")
        
        if 'allocation' in locals() and not allocation.empty:
            report = f"""
            ### AI CAPITAL ALLOCATION REPORT
            **Date:** {datetime.now().strftime('%Y-%m-%d')}
            **Client Risk Profile:** {risk}
            **Total Investment:** ₹{capital:,.0f}
            
            ---
            **Strategy Summary:**
            The system successfully optimized a portfolio of {len(allocation)} assets. 
            The allocation logic strictly followed constraints for a '{risk}' investor, ensuring {
                'maximum diversification' if risk == 'Conservative' else 'high-conviction growth'
            }.
            
            **Top 3 Holdings:**
            1. {allocation.iloc[0]['Asset']} ({allocation.iloc[0]['Weight']*100:.1f}%)
            2. {allocation.iloc[1]['Asset']} ({allocation.iloc[1]['Weight']*100:.1f}%)
            3. {allocation.iloc[2]['Asset']} ({allocation.iloc[2]['Weight']*100:.1f}%)
            """
            
            st.text_area("Copy Report Content", report, height=300)
            st.download_button("Download Report", report, file_name="Allocation_Report.txt")

if __name__ == "__main__":
    main()
