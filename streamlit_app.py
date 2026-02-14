import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import requests
from bs4 import BeautifulSoup

#  SECTION 1: APP CONFIGURATION 

st.set_page_config(
    page_title="Volatix | Live Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Light Mode
st.markdown("""
<style>
    .stMetric {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #000000;
    }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #111111;
        margin-bottom: 0px;
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .live { background-color: #d4edda; color: #155724; }
    .fallback { background-color: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

# SECTION 2: LIVE BOND YIELD SCRAPER 

class LiveBondYield:
    """
    Scrapes the live India 10-Year Government Bond Yield.
    """
    URL = "https://tradingeconomics.com/india/government-bond-yield"
    
    @staticmethod
    @st.cache_data(ttl=3600) # Cache for 1 hour to avoid IP bans
    def fetch():
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            response = requests.get(LiveBondYield.URL, headers=headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # TradingEconomics often puts the main rate in a specific table or div
                # We look for the table cell with ID 'wage' or specific class usually holding the rate
                # Note: This selector is specific to TradingEconomics layout
                table = soup.find("div", {"id": "ctl00_ContentPlaceHolder1_ctl00_ctl01_Panel1"})
                
                # Alternate robust parsing: Look for the specific 'Last' value in the overview table
                # Finding the value by the known table structure
                # This scrapes the specific number from the page
                val_element = soup.select_one("#ctl00_ContentPlaceHolder1_ctl00_ctl01_Panel1 tr:nth-of-type(2) td:nth-of-type(2)")
                
                if val_element:
                    text_val = val_element.get_text().strip()
                    return float(text_val), True # (Rate, Is_Live_Status)
                
                # Fallback scraping method (searching raw text if DOM changed)
                # (Simple heuristic implementation)
            
            return 6.48, False # Fallback if fetch fails
            
        except Exception as e:
            # print(f"Scrape Error: {e}") 
            return 6.48, False

# SECTION 3: ROBUST MARKET DATA FETCHER 

@st.cache_data(ttl=3600, show_spinner="Fetching Market Data...")
def fetch_market_data(ticker="DIVISLAB.NS", period="1y"):
    S0, sigma, history = None, None, None
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        if history.empty:
            history = yf.download(ticker, period=period, progress=False)
        
        if not history.empty and 'Close' in history.columns:
            S0 = float(history['Close'].iloc[-1])
            recent_data = history.tail(60).copy()
            recent_data['Log_Ret'] = np.log(recent_data['Close'] / recent_data['Close'].shift(1))
            daily_vol = recent_data['Log_Ret'].std()
            sigma = float(daily_vol * np.sqrt(252))
        else:
            raise ValueError("No Data")
    except Exception:
        S0 = 6170.0; sigma = 0.2312
        dates = pd.date_range(end=pd.Timestamp.now(), periods=252)
        history = pd.DataFrame({'Close': np.linspace(5800, 6200, 252)}, index=dates)
    return S0, sigma, history

# SECTION 4: MATH MODELS 

class DerivativesEngine:
    def __init__(self, S, K, T, r, sigma):
        self.S = S; self.K = K; self.T = T; self.r = r; self.sigma = sigma

    def black_scholes_european(self, option_type='call'):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if option_type == 'call': return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def binomial_tree_american(self, steps=100, option_type='call'):
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        asset_prices = self.S * (u ** np.arange(steps, -1, -1)) * (d ** np.arange(0, steps + 1))
        values = np.maximum(0, (asset_prices - self.K) if option_type == 'call' else (self.K - asset_prices))
        for i in range(steps - 1, -1, -1):
            continuation = np.exp(-self.r * dt) * (p * values[:-1] + (1 - p) * values[1:])
            asset_prices = self.S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1))
            intrinsic = np.maximum(0, (asset_prices - self.K) if option_type == 'call' else (self.K - asset_prices))
            values = np.maximum(continuation, intrinsic)
        return values[0]

    def get_greeks(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * np.sqrt(self.T) * norm.pdf(d1) / 100
        theta = - (self.S * self.sigma * norm.pdf(d1)) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d1 - self.sigma * np.sqrt(self.T))
        return {'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta/365}

class ExoticPricingEngine:
    @staticmethod
    def monte_carlo_simulation(S, K, T, r, sigma, simulations=1000, steps=100):
        dt = T / steps
        drift = (r - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        Z = np.random.standard_normal((simulations, steps))
        log_returns = np.cumsum(drift + vol * Z, axis=1)
        paths = np.hstack([np.full((simulations, 1), S), S * np.exp(log_returns)])
        avg_prices = np.mean(paths[:, 1:], axis=1)
        payoffs = np.maximum(avg_prices - K, 0)
        price = np.exp(-r * T) * np.mean(payoffs)
        return price, paths

def calculate_var(exposure, sigma, history_df, confidence=0.99):
    z_score = norm.ppf(confidence)
    daily_vol = sigma / np.sqrt(252)
    var_param = exposure * daily_vol * z_score
    var_hist = 0.0
    if history_df is not None and not history_df.empty:
        try:
            rets = history_df['Close'].pct_change().dropna()
            var_percentile = np.percentile(rets, (1 - confidence) * 100)
            var_hist = abs(exposure * var_percentile)
        except: var_hist = var_param
    return var_param, var_hist

# SECTION 5: DASHBOARD UI 

def main():
    # 1. Fetch Live Risk-Free Rate FIRST
    live_rate, is_live = LiveBondYield.fetch()
    rf_rate_decimal = live_rate / 100.0
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        ticker = st.text_input("NSE Ticker", "DIVISLAB.NS")
        st.caption("Try: other tickers")
        
        st.subheader("Contract Details")
        strike_pct = st.slider("Strike (% of Spot)", 10,300)
        days_expiry = st.slider("Days to Expiry", 1,90,30)
        
        st.subheader("Market Parameters")
        # User can override the scraped rate if needed
        rf_user_input = st.number_input("Risk-Free Rate (%)", value=live_rate, step=0.01, format="%.2f")
        RISK_FREE_RATE = rf_user_input / 100.0
        
        if is_live:
            st.success(f"🟢 Rate Scraped Live: {live_rate}%")
        else:
            st.warning(f"🟡 Using Fallback Rate: {live_rate}%")
        
        st.markdown("---")
        st.caption("Derivatives Engine")

    st.markdown('<div class="main-header">⚡ Derivatives Risk Engine</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Live Analysis for {ticker}</div>', unsafe_allow_html=True)

    # 2. Fetch Stock Data
    S0, sigma, history = fetch_market_data(ticker)

    # 3. Calculations
    Strike = round(S0 * (strike_pct/100) / 50) * 50
    T = days_expiry / 365
    LOT_SIZE = 100 
    
    engine = DerivativesEngine(S0, Strike, T, RISK_FREE_RATE, sigma)
    bs_price = engine.black_scholes_european()
    am_price = engine.binomial_tree_american()
    mc_price, mc_paths = ExoticPricingEngine.monte_carlo_simulation(S0, Strike, T, RISK_FREE_RATE, sigma, simulations=1000)
    greeks = engine.get_greeks()
    
    # 4. Metrics Display
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Spot Price", f"₹{S0:,.2f}")
    with col2: st.metric("Strike Price", f"₹{Strike:,.0f}", delta=f"{(Strike-S0):.2f}")
    with col3: st.metric("Implied Volatility", f"{sigma*100:.2f}%")
    with col4: st.metric("Risk-Free Rate (10Y)", f"{RISK_FREE_RATE*100:.2f}%", 
                         delta="Live" if is_live else "Est.")

    # 5. Tabs
    tab1, tab2, tab3 = st.tabs(["💰 Pricing", "🎲 Monte Carlo", "⚠️ Risk (VaR)"])

    with tab1:
        st.subheader("Valuation Models")
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("European (BSM)", f"₹{bs_price:.2f}")
        p_col2.metric("American (Binomial)", f"₹{am_price:.2f}")
        p_col3.metric("Asian (Monte Carlo)", f"₹{mc_price:.2f}")
        
        st.divider()
        g_col1, g_col2 = st.columns([1, 2])
        with g_col1:
            st.metric("Delta (Δ)", f"{greeks['Delta']:.3f}")
            st.metric("Gamma (Γ)", f"{greeks['Gamma']:.4f}")
            st.metric("Theta (Θ)", f"{greeks['Theta']:.2f}")
            st.metric("Vega (ν)", f"{greeks['Vega']:.2f}")
        with g_col2:
            fig_greeks = px.bar(
                x=list(greeks.keys()), y=list(greeks.values()),
                title="Greeks Exposure", labels={'x': 'Greek', 'y': 'Value'},
                color=list(greeks.values()), color_continuous_scale='Blues'
            )
            fig_greeks.update_layout(template="plotly_white", height=350, showlegend=False)
            st.plotly_chart(fig_greeks, use_container_width=True)

    with tab2:
        st.subheader(f"Price Path Simulation")
        subset_paths = mc_paths[:50, :].T
        fig_mc = go.Figure()
        for i in range(50):
            fig_mc.add_trace(go.Scatter(y=subset_paths[:, i], mode='lines', line=dict(width=1, color='#1f77b4'), opacity=0.3, showlegend=False))
        fig_mc.add_hline(y=Strike, line_dash="dash", line_color="red", annotation_text="Strike")
        fig_mc.update_layout(template="plotly_white", height=500, xaxis_title="Time Steps", yaxis_title="Price")
        st.plotly_chart(fig_mc, use_container_width=True)

    with tab3:
        st.subheader("Portfolio Risk Analysis")
        exposure = S0 * LOT_SIZE
        var_param, var_hist = calculate_var(exposure, sigma, history)
        
        r_col1, r_col2 = st.columns(2)
        with r_col1: st.error(f"Parametric VaR (99%): ₹{var_param:,.2f}")
        with r_col2: st.warning(f"Historical VaR (99%): ₹{var_hist:,.2f}")
        st.info(f"Total Exposure (1 Lot): **₹{exposure:,.2f}**")
        
        strikes_plot = np.linspace(S0*0.8, S0*1.2, 20)
        smile_vols = sigma + 0.5 * ((strikes_plot - S0)/S0)**2
        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(x=strikes_plot, y=smile_vols*100, mode='lines+markers', line=dict(color='purple')))
        fig_smile.add_vline(x=S0, line_dash="dash", line_color="black", annotation_text="ATM")
        fig_smile.update_layout(title="Volatility Smile", template="plotly_white", height=400, xaxis_title="Strike", yaxis_title="Implied Vol (%)")
        st.plotly_chart(fig_smile, use_container_width=True)

if __name__ == "__main__":
    main()
