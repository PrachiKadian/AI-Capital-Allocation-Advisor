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
# 1. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AlphaAlloc | AI Capital Allocation System",
    layout="wide",
    page_icon="🇮🇳",
    initial_sidebar_state="expanded"
)

# --- Constants & Tickers ---
RISK_FREE_RATE = 0.070  # India 10Y Bond Yield approx
BENCHMARK = "^NSEI"     # Nifty 50

# Curated List of Indian Assets for Demo
ASSETS = {
    "Stocks": [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 
        'ICICIBANK.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS',
        'LT.NS', 'HINDUNILVR.NS', 'AXISBANK.NS', 'MARUTI.NS'
    ],
    "Indices": ['^NSEI', '^NSEBANK', '^BSESN'],
    "Mutual_Funds": ['0P0000XVAA.BO', '0P0000XW8F.BO', '0P0000XVTR.BO'] # Proxies: HDFC Top 100, SBI Bluechip, etc.
}

# ==========================================
# 2. DATA INGESTION ENGINE (Robust)
# ==========================================
@st.cache_data(ttl=3600)  # Cache data for 1 hour to prevent API spam
def fetch_historical_data(tickers, period="2y"):
    """
    Robust data fetcher that handles yfinance MultiIndex structures.
    """
    if not tickers:
        return pd.DataFrame()
    
    try:
        data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, progress=False)
        
        # Flatten MultiIndex if necessary
        prices = pd.DataFrame()
        
        for t in tickers:
            try:
                if len(tickers) == 1:
                    # Single ticker structure
                    if 'Close' in data.columns:
                        prices[t] = data['Close']
                    else:
                        prices[t] = data
                else:
                    # Multi-ticker structure
                    if t in data.columns.levels[0]:
                        prices[t] = data[t]['Close']
            except KeyError:
                continue
                
        prices.dropna(how='all', inplace=True)
        return prices
    except Exception as e:
        st.error(f"Data Fetch Error: {str(e)}")
        return pd.DataFrame()

def fetch_live_metrics(ticker):
    """Get latest price and daily change."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2d")
        if len(hist) >= 1:
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Open'].iloc[-1]
            change_pct = ((current_price - prev_close) / prev_close) * 100
            return current_price, change_pct
        return 0.0, 0.0
    except:
        return 0.0, 0.0

# ==========================================
# 3. ANALYTICS & AI CORE
# ==========================================
def calculate_risk_metrics(price_series):
    """Computes Volatility, Sharpe, Max Drawdown."""
    returns = price_series.pct_change().dropna()
    
    # Metrics
    annual_vol = returns.std() * np.sqrt(252)
    total_ret = (price_series.iloc[-1] / price_series.iloc[0]) - 1
    sharpe = (total_ret - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else 0
    
    # Max Drawdown
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    return total_ret, annual_vol, sharpe, max_dd

def ml_trend_forecast(price_series, days=180):
    """
    Linear Regression Model for Trend Projection.
    Output: Projected Return %, R-Squared Score, Plot Data
    """
    df = price_series.to_frame(name='Close').reset_index()
    df['Time'] = np.arange(len(df))
    
    X = df[['Time']]
    y = df['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast
    future_X = np.arange(len(df), len(df) + days).reshape(-1, 1)
    future_prices = model.predict(future_X)
    
    # Calc Return
    current_price = y.iloc[-1]
    target_price = future_prices[-1]
    exp_return = ((target_price - current_price) / current_price) * 100
    r2 = model.score(X, y)
    
    return exp_return, r2, future_prices

def monte_carlo_sim(weights, mean_returns, cov_matrix, initial_portfolio, days=252, simulations=1000):
    """
    Monte Carlo Simulation using Geometric Brownian Motion (GBM).
    """
    # Portfolio stats
    port_mean = np.sum(weights * mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Simulation
    simulation_df = pd.DataFrame()
    
    for x in range(simulations):
        # GBM Formula: S_t = S_0 * exp((mu - 0.5*sigma^2)t + sigma * Z)
        # We simulate daily returns
        daily_vol = port_vol / np.sqrt(252)
        daily_ret = port_mean / 252
        
        random_shocks = np.random.normal(0, 1, days)
        price_series = [initial_portfolio]
        
        for shock in random_shocks:
            price = price_series[-1] * np.exp((daily_ret - 0.5 * daily_vol**2) + daily_vol * shock)
            price_series.append(price)
            
        simulation_df[x] = price_series
        
    return simulation_df

def optimize_portfolio_weights(tickers, risk_constraint):
    """
    Mean-Variance Optimization (Markowitz).
    """
    df = fetch_historical_data(tickers, period="1y")
    returns = df.pct_change().dropna()
    
    mu = returns.mean() * 252
    sigma = returns.cov() * 252
    num_assets = len(tickers)
    
    # Constraints & Bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.05, 0.4) for _ in range(num_assets)) # Min 5%, Max 40% per asset
    
    def neg_sharpe(weights):
        p_ret = np.sum(weights * mu)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        return -(p_ret - RISK_FREE_RATE) / p_vol
    
    # Optimization
    init_guess = [1./num_assets] * num_assets
    result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x, df.columns

# ==========================================
# 4. FRONTEND UI LAYOUT
# ==========================================
def main():
    # --- Sidebar ---
    st.sidebar.title("⚙️ Configuration")
    user_capital = st.sidebar.number_input("Capital Available (₹)", 50000, 10000000, 500000, step=10000)
    user_horizon = st.sidebar.selectbox("Investment Horizon", ["1 Year", "3 Years", "5 Years"])
    user_risk = st.sidebar.select_slider("Risk Tolerance", options=["Conservative", "Moderate", "Aggressive"])
    
    st.sidebar.info(f"**Profile:** {user_risk} | **Horizon:** {user_horizon}")
    st.sidebar.markdown("---")
    st.sidebar.caption("MSc Finance Final Project | KD")

    # --- Header ---
    st.title("🇮🇳 AlphaAlloc: AI-Powered Portfolio Advisor")
    st.markdown("Advanced Capital Allocation System for Indian Equities & Mutual Funds.")
    
    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Market Intel", 
        "📈 Stock Analysis", 
        "💰 Mutual Funds", 
        "🤖 Portfolio AI", 
        "⚡ Stress Test (MC)", 
        "📝 Report"
    ])

    # ---------------- TAB 1: MARKET INTEL ----------------
    with tab1:
        st.subheader("Market Regime Awareness Layer")
        
        # Live Ticker Strip
        c1, c2, c3 = st.columns(3)
        nifty_p, nifty_chg = fetch_live_metrics(BENCHMARK)
        bank_p, bank_chg = fetch_live_metrics("^NSEBANK")
        
        c1.metric("NIFTY 50", f"₹{nifty_p:,.2f}", f"{nifty_chg:.2f}%")
        c2.metric("BANK NIFTY", f"₹{bank_p:,.2f}", f"{bank_chg:.2f}%")
        c3.metric("Market Status", "Open" if datetime.now().hour < 16 else "Closed", border=True)
        
        # Regime Classification
        st.divider()
        st.markdown("#### 🧠 AI Regime Classification")
        
        # Simple Regime Logic (Academic Proxy)
        ma_df = fetch_historical_data([BENCHMARK], period="1y")
        if not ma_df.empty:
            sma_50 = ma_df[BENCHMARK].rolling(50).mean().iloc[-1]
            sma_200 = ma_df[BENCHMARK].rolling(200).mean().iloc[-1]
            curr_price = ma_df[BENCHMARK].iloc[-1]
            
            if curr_price > sma_50 > sma_200:
                regime = "BULLISH (Aggressive Allocation)"
                color = "green"
            elif curr_price < sma_50 < sma_200:
                regime = "BEARISH (Defensive Allocation)"
                color = "red"
            else:
                regime = "SIDEWAYS / VOLATILE (Neutral)"
                color = "orange"
                
            st.markdown(f"**Detected Regime:** :{color}[{regime}]")
            st.caption("Logic: Price vs 50DMA vs 200DMA Golden/Death Cross analysis.")
            
            # Chart
            fig = px.line(ma_df, y=BENCHMARK, title="NIFTY 50 Trend Analysis")
            fig.add_scatter(x=ma_df.index, y=ma_df[BENCHMARK].rolling(50).mean(), name="50 DMA", line=dict(color='orange'))
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- TAB 2: EQUITY ANALYSIS ----------------
    with tab2:
        st.subheader("Deep Dive: Stock Intelligence")
        selected_ticker = st.selectbox("Select Asset", ASSETS["Stocks"])
        
        if selected_ticker:
            with st.spinner("Fetching Data & Running ML Models..."):
                stock_df = fetch_historical_data([selected_ticker], period="3y")
                
                if not stock_df.empty:
                    series = stock_df[selected_ticker]
                    
                    # 1. Metrics
                    tot_ret, vol, sharpe, dd = calculate_risk_metrics(series)
                    
                    # 2. ML Forecast
                    exp_ret_6m, r2, future_vals = ml_trend_forecast(series)
                    
                    # Display
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Annual Volatility", f"{vol*100:.1f}%", help="Standard Deviation of Returns")
                    c2.metric("Sharpe Ratio", f"{sharpe:.2f}", help="Risk-Adjusted Return")
                    c3.metric("Max Drawdown", f"{dd*100:.1f}%", help="Worst fall from peak")
                    c4.metric("AI Forecast (6M)", f"{exp_ret_6m:.1f}%", f"R²: {r2:.2f}")
                    
                    # Visuals
                    chart_col, info_col = st.columns([3, 1])
                    with chart_col:
                        fig = px.line(series, title=f"{selected_ticker} Historical Price")
                        
                        # Add Forecast Line (Visual Hack)
                        last_date = series.index[-1]
                        future_dates = [last_date + timedelta(days=i) for i in range(len(future_vals))]
                        fig.add_scatter(x=future_dates, y=future_vals, name="AI Projection (Linear Reg)", line=dict(dash='dot', color='green'))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with info_col:
                        st.info("ℹ️ **Model Logic:** Uses Linear Regression on time-series data to project momentum. R-squared indicates how well the trend fits history.")

    # ---------------- TAB 3: MUTUAL FUNDS ----------------
    with tab3:
        st.subheader("Mutual Fund Screener")
        st.markdown("*Note: Direct API access to Indian MF NAVs is limited. Showing Index Proxies for Academic Demonstration.*")
        
        mf_tickers = ASSETS["Mutual_Funds"]
        mf_data = fetch_historical_data(mf_tickers, period="1y")
        
        if not mf_data.empty:
            metrics_list = []
            for mf in mf_tickers:
                t_ret, t_vol, t_sharpe, t_dd = calculate_risk_metrics(mf_data[mf])
                metrics_list.append({
                    "Fund Ticker": mf,
                    "Annual Return": f"{t_ret*100:.1f}%",
                    "Volatility": f"{t_vol*100:.1f}%",
                    "Sharpe Ratio": round(t_sharpe, 2)
                })
            
            st.dataframe(pd.DataFrame(metrics_list), use_container_width=True)
            
            # Comparative Chart
            norm_df = mf_data / mf_data.iloc[0] * 100
            st.plotly_chart(px.line(norm_df, title="Normalized Performance (Base=100)"), use_container_width=True)

    # ---------------- TAB 4: PORTFOLIO OPTIMIZATION ----------------
    with tab4:
        st.subheader("🤖 AI Portfolio Builder (Markowitz Model)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Construction Parameters**")
            selected_universe = st.multiselect("Select Assets for Portfolio", ASSETS["Stocks"], default=ASSETS["Stocks"][:5])
            
            if st.button("🚀 Optimize Allocation"):
                if len(selected_universe) < 2:
                    st.error("Select at least 2 assets.")
                else:
                    with st.spinner("Running Quadratic Optimization..."):
                        weights, asset_names = optimize_portfolio_weights(selected_universe, user_risk)
                        
                        # Results
                        st.session_state['opt_weights'] = weights
                        st.session_state['opt_assets'] = asset_names
                        st.session_state['run_opt'] = True
        
        with col2:
            if st.session_state.get('run_opt'):
                weights = st.session_state['opt_weights']
                assets = st.session_state['opt_assets']
                
                # Pie Chart
                df_alloc = pd.DataFrame({"Asset": assets, "Weight": weights})
                df_alloc['Value'] = df_alloc['Weight'] * user_capital
                
                fig = px.pie(df_alloc, values='Weight', names='Asset', title="Optimal Efficient Frontier Allocation", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
                
                # Allocation Table
                st.dataframe(df_alloc.style.format({"Weight": "{:.1%}", "Value": "₹{:.2f}"}))

    # ---------------- TAB 5: SCENARIO & MONTE CARLO ----------------
    with tab5:
        st.subheader("⚡ Stress Testing & Monte Carlo Simulation")
        
        if st.session_state.get('run_opt'):
            weights = st.session_state['opt_weights']
            assets = st.session_state['opt_assets']
            
            # Get data for Covariance Matrix
            data = fetch_historical_data(list(assets), period="1y")
            returns = data.pct_change().dropna()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # A. STATIC SCENARIOS
            st.markdown("#### A. Static Scenario Analysis")
            scenarios = {
                "2008 Crash (-50% Equity)": -0.50,
                "COVID Correction (-30%)": -0.30,
                "Normal Correction (-10%)": -0.10,
                "Bull Run (+20%)": 0.20
            }
            
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                sel_scen = st.selectbox("Choose Historical Scenario", list(scenarios.keys()))
                shock = scenarios[sel_scen]
                curr_val = user_capital
                new_val = curr_val * (1 + shock)
                st.metric("Portfolio Value Impact", f"₹{new_val:,.0f}", f"{shock*100}%", delta_color="normal")
            
            # B. MONTE CARLO
            st.divider()
            st.markdown("#### B. Monte Carlo Simulation (1000 Future Paths)")
            st.caption("Simulating 1 Year of potential returns using Geometric Brownian Motion.")
            
            if st.button("Run Monte Carlo"):
                with st.spinner("Simulating 1,000 parallel market universes..."):
                    sim_df = monte_carlo_sim(weights, mean_returns, cov_matrix, user_capital)
                    
                    # Plot
                    fig_mc = go.Figure()
                    # Plot first 50 traces for speed/visuals
                    for c in sim_df.columns[:50]:
                        fig_mc.add_trace(go.Scatter(y=sim_df[c], mode='lines', opacity=0.1, showlegend=False, line=dict(color='blue')))
                    
                    # Add Mean path
                    fig_mc.add_trace(go.Scatter(y=sim_df.mean(axis=1), mode='lines', name='Average Path', line=dict(color='red', width=3)))
                    
                    fig_mc.update_layout(title="Projected Portfolio Value (1 Year)", xaxis_title="Trading Days", yaxis_title="Portfolio Value (₹)")
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    # Stats
                    final_vals = sim_df.iloc[-1]
                    var_95 = np.percentile(final_vals, 5)
                    st.error(f"**Value at Risk (95% Confidence):** You will likely not lose more than ₹{user_capital - var_95:,.2f} in a bad year.")
        else:
            st.warning("⚠️ Please run the Optimization in the 'Portfolio AI' tab first to define weights.")

    # ---------------- TAB 6: REPORTING ----------------
    with tab6:
        st.subheader("📝 Investment Memorandum")
        
        if st.session_state.get('run_opt'):
            assets = st.session_state['opt_assets']
            weights = st.session_state['opt_weights']
            
            report_text = f"""
            ### INVESTMENT STRATEGY REPORT
            **Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
            **Client Capital:** ₹{user_capital:,.2f}
            **Risk Profile:** {user_risk}
            
            ---
            **1. Allocation Strategy**
            The system utilized Mean-Variance Optimization to maximize the Sharpe Ratio. 
            The selected portfolio consists of {len(assets)} assets with weights optimized to reduce idiosyncratic risk.
            
            **2. Top Holdings:**
            """
            for a, w in zip(assets, weights):
                if w > 0.01:
                    report_text += f"- **{a}:** {w*100:.2f}%\n"
            
            report_text += """
            \n**3. Risk Statement**
            This portfolio is subject to market risks. The Monte Carlo simulation indicates a 95% confidence that losses will not exceed calculated VaR limits.
            """
            
            st.markdown(report_text)
            st.download_button("📥 Download Report as PDF", data=report_text, file_name="Investment_Memo.txt")
        else:
            st.info("Generate a portfolio to view the report.")

if __name__ == "__main__":
    main()
