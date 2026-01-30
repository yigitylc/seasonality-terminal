# Seasonality Terminal - Streamlit App

Bloomberg-style seasonality heatmap dashboard with quant analysis panels for any ticker.

## Features

- **Seasonality Heatmap**: Monthly returns/levels with trailing averages and per-year rows
- **VIX Support**: Automatically switches to Δ points or level mode for ^VIX
- **Stability Metrics**: Mean, median, hit rate, downside frequency, IQR by month
- **Rolling Regime**: Track how a single month's seasonality evolves over rolling windows
- **Distribution Boxplots**: Monthly return distributions with optional jitter overlay
- **Dark Theme**: Bloomberg-inspired dark layout across all charts

## Local Run (Windows PowerShell)

```powershell
cd streamlit_apps/seasonality-terminal

# Create virtual environment
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at http://localhost:8501

## Streamlit Community Cloud Deployment

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set the main file path to `app.py`
5. Deploy

No secrets required - the app uses public data sources (yfinance).

## Data Sources

- **Market Data**: yfinance (Yahoo Finance, adjusted prices via auto_adjust=True)
