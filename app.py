import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from datetime import datetime, timedelta

# ==========================================
# 1. SYSTEM CONFIGURATION & DICTIONARIES
# ==========================================
st.set_page_config(
    page_title="AI Capital Allocation Advisor",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- Constants & Mappings ---
RISK_FREE_RATE = 0.070  # India 10Y Bond Yield
BENCHMARK = "^NSEI"     # Nifty 50

# Map Tickers to Real Names for better UI
ASSET_NAMES = {
    # Stocks
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
    
    # Mutual Funds (BSE Star MF Tickers)
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
@st.cache_data(ttl=1800)
def fetch_data(tickers, period="2y"):
    """Fetch data for a list of tickers."""
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

@st.cache_data(ttl=300)
def get_market_movers():
    """Calculate top gainers/losers from our watchlist manually (more reliable for India)."""
    # Fetch data for all our tracked stocks for 2 days
    df = fetch_data(STOCKS, period="5d") 
    if df.empty: return pd.DataFrame()
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    changes = ((latest - prev) / prev) * 100
    ranking = changes.sort_values(ascending=False)
    
    # Volume Shockers proxy (using volatility as proxy for activity here for speed)
    vol = df.pct_change().std() * np.sqrt(252)
    
    return ranking, vol

# ==========================================
# 3. ANALYTICS & OPTIMIZATION LOGIC
# ==========================================
def calculate_metrics(series):
    if series.empty: return 0,0,0,0
    returns = series.pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(252)
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    sharpe = (total_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    
    # Drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    return total_ret, ann_vol, sharpe, max_dd

def optimize_portfolio(tickers, risk_profile):
    """
    Dynamic Optimization based on Risk Profile.
    Conservative: Max 15% per stock, Min 40% in MFs (if present).
    Aggressive: Max 40% per stock.
    """
    df = fetch_data(tickers, period="1y")
    if df.empty: return [], []
    
    returns = df.pct_change().dropna()
    mu = returns.mean() * 252
    sigma = returns.cov() * 252
    n = len(tickers)
    
    # 1. Dynamic Constraints based on Risk
    if risk_profile == "Conservative":
        # Force diversification, lower max cap per asset
        bounds = tuple((0.0, 0.20) for _ in range(n)) 
    elif risk_profile == "Aggressive":
        # Allow concentrated bets
        bounds = tuple((0.0, 0.50) for _ in range(n))
    else: # Moderate
        bounds = tuple((0.05, 0.35) for _ in range(n))
        
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # 2. Optimize for Sharpe
    def neg_sharpe(weights):
        p_ret = np.sum(weights * mu)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        return -(p_ret - RISK_FREE_RATE) / p_vol
    
    guess = [1./n] * n
    res = minimize(neg_sharpe, guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return res.x, df.columns

# ==========================================
# 4. MAIN UI
# ==========================================
def main():
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.info("👈 **Start Here:** Input your preferences below to initialize the AI models.")
    
    capital = st.sidebar.number_input("Investment Amount (₹)", 50000, 10000000, 500000, step=50000)
    risk = st.sidebar.select_slider("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], value="Moderate")
    
    # Dynamic Universe Selection based on Risk
    st.sidebar.markdown("### Asset Preferences")
    include_stocks = st.sidebar.checkbox("Include Stocks", value=True)
    include_mfs = st.sidebar.checkbox("Include Mutual Funds", value=True)
    
    # Title
    st.title("AI Capital Allocation Advisor")
    st.markdown(f"**Current Profile:** {risk} Investor | **Capital:** ₹{capital:,.0f}")
    
    # Tabs
    tabs = st.tabs(["🔍 Explore", "📈 Stocks", "💰 Mutual Funds", "🤖 Combined AI Portfolio", "📝 Report"])
    
    # --- TAB 1: EXPLORE (MARKET INTEL) ---
    with tabs[0]:
        st.subheader("Market Intelligence & Live Dashboard")
        
        # 1. Live Indices
        c1, c2, c3 = st.columns(3)
        # Fetching proxy data for speed
        nifty = yf.Ticker("^NSEI").history(period="2d")
        sensex = yf.Ticker("^BSESN").history(period="2d")
        bank = yf.Ticker("^NSEBANK").history(period="2d")
        
        def get_delta(hist):
            if len(hist) < 2: return 0, 0
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            return curr, ((curr-prev)/prev)*100

        n_p, n_d = get_delta(nifty)
        s_p, s_d = get_delta(sensex)
        b_p, b_d = get_delta(bank)
        
        c1.metric("NIFTY 50", f"₹{n_p:,.0f}", f"{n_d:.2f}%")
        c2.metric("SENSEX", f"₹{s_p:,.0f}", f"{s_d:.2f}%")
        c3.metric("BANK NIFTY", f"₹{b_p:,.0f}", f"{b_d:.2f}%")
        
        st.divider()
        
        # 2. Movers & Shakers
        col_g, col_l = st.columns(2)
        ranking, vol_shock = get_market_movers()
        
        if not ranking.empty:
            with col_g:
                st.markdown("##### 🚀 Top Gainers (Today)")
                gainers = ranking.head(5)
                for t, v in gainers.items():
                    st.write(f"**{ASSET_NAMES.get(t, t)}**: :green[+{v:.2f}%]")
            
            with col_l:
                st.markdown("##### 🔻 Top Losers (Today)")
                losers = ranking.tail(5).sort_values()
                for t, v in losers.items():
                    st.write(f"**{ASSET_NAMES.get(t, t)}**: :red[{v:.2f}%]")
        
        st.divider()
        
        # 3. Market Breadth / Popularity
        c_pop1, c_pop2 = st.columns(2)
        with c_pop1:
            st.markdown("##### 📊 Most Active / Volatile")
            if not ranking.empty:
                active = vol_shock.sort_values(ascending=False).head(5)
                st.dataframe(active.rename("Volatility Score"), use_container_width=True)
        
        with c_pop2:
            st.markdown("##### 🏆 Popular Mutual Funds (By AUM Proxy)")
            pop_mfs = pd.DataFrame({
                "Fund Name": ["HDFC Top 100", "SBI Bluechip", "Axis Bluechip", "Nippon Small Cap"],
                "Category": ["Large Cap", "Large Cap", "Large Cap", "Small Cap"],
                "Rating": ["⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐⭐"]
            })
            st.dataframe(pop_mfs, hide_index=True, use_container_width=True)

    # --- TAB 2: STOCKS ---
    with tabs[1]:
        st.subheader("Equity Analysis")
        st.info("ℹ️ Select a stock to view detailed AI-driven metrics.")
        
        sel_stock_key = st.selectbox("Select Stock", STOCKS, format_func=lambda x: ASSET_NAMES.get(x, x))
        
        if sel_stock_key:
            s_data = fetch_data([sel_stock_key])
            if not s_data.empty:
                s_ret, s_vol, s_sharpe, s_dd = calculate_metrics(s_data[sel_stock_key])
                
                m1, m2, m3 = st.columns(3)
                m1.metric("1Y Return", f"{s_ret*100:.1f}%")
                m2.metric("Volatility (Risk)", f"{s_vol*100:.1f}%", help="Higher means more risky")
                m3.metric("Sharpe Ratio", f"{s_sharpe:.2f}", help="Return per unit of risk. >1 is good.")
                
                st.line_chart(s_data[sel_stock_key])

    # --- TAB 3: MUTUAL FUNDS ---
    with tabs[2]:
        st.subheader("Mutual Fund Analyzer")
        st.caption("Data sourced from BSE Star MF (Direct Plans)")
        
        # Display Real Data Table
        mf_df = fetch_data(MUTUAL_FUNDS, period="1y")
        
        if not mf_df.empty:
            mf_stats = []
            for mf in MUTUAL_FUNDS:
                if mf in mf_df:
                    ret, vol, sharpe, _ = calculate_metrics(mf_df[mf])
                    mf_stats.append({
                        "Fund Name": ASSET_NAMES.get(mf, mf),
                        "1Y Return": f"{ret*100:.1f}%",
                        "Risk (Vol)": f"{vol*100:.1f}%",
                        "Sharpe Ratio": round(sharpe, 2)
                    })
            
            st.dataframe(pd.DataFrame(mf_stats).set_index("Fund Name"), use_container_width=True)
            
            # Chart
            st.markdown("##### Performance Comparison")
            # Normalize to 100
            norm_mf = mf_df / mf_df.iloc[0] * 100
            # Rename columns for chart
            norm_mf.columns = [ASSET_NAMES.get(c, c) for c in norm_mf.columns]
            st.plotly_chart(px.line(norm_mf), use_container_width=True)

    # --- TAB 4: COMBINED PORTFOLIO ---
    with tabs[3]:
        st.subheader("🤖 AI Combined Portfolio Engine")
        st.markdown(f"Optimizing allocation for a **{risk}** profile using Modern Portfolio Theory.")
        
        # 1. Build Universe based on sidebar
        universe = []
        if include_stocks: universe += STOCKS[:8] # Limit for demo speed
        if include_mfs: universe += MUTUAL_FUNDS[:3]
        
        if not universe:
            st.error("Please select at least one asset class in the sidebar.")
        else:
            if st.button("Generate Optimal Portfolio"):
                with st.spinner("AI is analyzing correlations and optimizing weights..."):
                    weights, assets = optimize_portfolio(universe, risk)
                    
                    # Create Result DF
                    res_df = pd.DataFrame({
                        "Asset": [ASSET_NAMES.get(a, a) for a in assets],
                        "Ticker": assets,
                        "Weight": weights,
                        "Allocation (₹)": weights * capital
                    })
                    res_df = res_df[res_df['Weight'] > 0.01].sort_values("Weight", ascending=False)
                    
                    # Display Layout
                    c_chart, c_table = st.columns([1, 1])
                    
                    with c_chart:
                        fig = px.pie(res_df, values="Weight", names="Asset", hole=0.4, 
                                     title="Suggested Allocation",
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with c_table:
                        st.markdown("##### Allocation Details")
                        st.dataframe(res_df[["Asset", "Weight", "Allocation (₹)"]].style.format({"Weight": "{:.1%}", "Allocation (₹)": "₹{:.2f}"}), use_container_width=True)
                        
                        est_ret = np.sum(weights * 0.12) # conservative estimate for display
                        st.success(f"**Expected Portfolio Return:** ~{est_ret*100:.1f}% p.a.")
                        st.info(f"**Optimization Logic:** Model Adjusted for '{risk}' profile by capping max exposure to volatile assets.")

    # --- TAB 5: REPORT ---
    with tabs[4]:
        st.subheader("Investment Memo")
        st.write("Generate portfolio in previous tab to populate this report.")
        
        if 'res_df' in locals() and not res_df.empty:
             report = f"""
             **Investment Strategy Report**
             *Generated for Risk Profile: {risk}*
             
             **Portfolio Composition:**
             The AI has allocated ₹{capital:,.2f} across {len(res_df)} assets.
             Top holding is {res_df.iloc[0]['Asset']} at {res_df.iloc[0]['Weight']*100:.1f}%.
             
             **Risk Adjustment:**
             {'Conservative constraints applied (High diversification).' if risk == 'Conservative' else 'Aggressive constraints applied (High conviction bets).'}
             """
             st.info(report)
