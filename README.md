# ⚡ Volatix: Derivatives & Risk Engine

> **A real-time Finance Dashboard designed to price Indian Equity Derivatives (NSE), analyze Volatility Smiles, and quantify Portfolio Risk (VaR) using live market data.**

---

## 🚀 Overview

**Volatix** is an advanced risk engineering tool built for the Indian markets. Unlike static calculators, it connects to live data sources to provide real-time decision support for traders and risk managers.

It solves three critical problems:
1.  **Valuation:** Is the option fairly priced? (Comparing BSM vs. Binomial Models).
2.  **Risk:** What is the maximum I could lose tomorrow? (Value at Risk - VaR).
3.  **Sensitivity:** How will my portfolio react to market shocks? (Real-time Greeks).

---

## ✨ Key Features

### 1. 🟢 Live Rate Scraping (Bot)
- Automatically scrapes the **India 10-Year Government Bond Yield** from *Trading Economics* to use as a dynamic Risk-Free Rate.
- Includes a fallback mechanism to ensure the engine never crashes if the website is down.

### 2. 💰 Dual-Pricing Architecture
- **Black-Scholes-Merton (European):** Standard valuation model for NSE options.
- **Cox-Ross-Rubinstein (Binomial Tree):** Quantifies the "Early Exercise Premium" (American style) to identify arbitrage opportunities.

### 3. 🎲 Monte Carlo Simulations
- Runs **1,000+ stochastic price paths** to price path-dependent Exotic Options (Asian Options).
- Visualizes the "Cone of Uncertainty" for future price movements.

### 4. ⚠️ Advanced Risk Metrics
- **Parametric VaR (99%):** Estimates worst-case daily loss using normal distribution theory.
- **Historical VaR:** Backtests against actual past returns to catch "Fat Tail" risks.
- **Volatility Smile:** Simulates the market skew for Out-of-the-Money (OTM) options.
---

## 📊 Mathematical Models Used

| Component | Model Implementation | Purpose |
| :--- | :--- | :--- |
| **Asset Dynamics** | Geometric Brownian Motion (GBM) | Simulating future stock price paths. |
| **Option Pricing** | Black-Scholes PDE | Closed-form solution for European Calls/Puts. |
| **Early Exercise** | CRR Binomial Tree | Backward induction for American Options. |
| **Risk (VaR)** | Parametric Variance-Covariance | $VaR = \text{Position} \times \sigma \times Z_{\alpha}$ |

---
## 🛠️ Tech Stack

* **Languages:** `Python` 
* **Data Pipeline:** `yfinance` (NSE Data), `BeautifulSoup` (Web Scraping)
* **Computation:** `NumPy` (Vectorized Math), `SciPy` (Statistical Models)
* **Visualization:** `Plotly` (Interactive Charts & Heatmaps)


## ⚙️ Installation 

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Akshat-Singh-Kshatriya/volatix-live-engine.git
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run project.py
    ```
