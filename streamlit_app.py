"""
Interactive Stock Price Forecasting Interface with Portfolio Analysis

Streamlit application for LSTM-based stock price forecasting with
portfolio-level predictions and risk metrics.

Author: Ansel Scheibel
Date: December 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import the forecaster
try:
    from lstm_stock_forecaster import StockPriceForecaster
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    st.error("Model not found. Make sure lstm_stock_forecaster.py is in the same directory.")


# Page configuration
st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 LSTM Stock Price Forecaster")
st.markdown("""
Forecast stock prices using Long Short-Term Memory (LSTM) neural networks with technical indicators.
""")

# Create tabs for Single Stock and Portfolio Analysis
tab1, tab2 = st.tabs(["📊 Single Stock Analysis", "💼 Portfolio Forecast"])

# ============================================================================
# TAB 1: SINGLE STOCK ANALYSIS (Original functionality)
# ============================================================================
with tab1:
    # Sidebar for configuration
    st.sidebar.header("Single Stock Configuration")

    # Stock selection
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="Enter any valid stock ticker")

    # Date range
    st.sidebar.subheader("Data Range")
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
    start_date = st.sidebar.date_input("Start Date", value=end_date - timedelta(days=3*365))

    # Hyperparameters
    st.sidebar.subheader("Model Hyperparameters")
    sequence_length = st.sidebar.slider("Sequence Length (days)", 30, 120, 60, 10)
    lstm_units = st.sidebar.slider("LSTM Units", 25, 200, 50, 25)
    num_layers = st.sidebar.slider("Number of LSTM Layers", 1, 4, 2)
    dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1)
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 50, 10)

    # Forecast settings
    st.sidebar.subheader("Forecast Settings")
    forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30, 7)

    # Technical indicators toggle
    use_indicators = st.sidebar.checkbox("Use Technical Indicators", value=True)

    # Train button
    train_button = st.sidebar.button("Train Model", type="primary")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Model Status")
        status_placeholder = st.empty()

        st.subheader("Price History and Predictions")
        chart_placeholder = st.empty()

    with col2:
        st.subheader("Performance Metrics")
        metrics_placeholder = st.empty()

        st.subheader("Model Configuration")
        config_placeholder = st.empty()

    # Initialize session state
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = None

    # Display current configuration
    with config_placeholder.container():
        config_data = {
            "Parameter": ["Ticker", "Sequence Length", "LSTM Units", "Layers", "Dropout", "Epochs"],
            "Value": [ticker, sequence_length, lstm_units, num_layers, dropout, epochs]
        }
        st.dataframe(pd.DataFrame(config_data), hide_index=True)

    # Training logic
    if train_button and HAS_MODEL:
        status_placeholder.info("🔄 Training model... This may take a few minutes.")

        try:
            forecaster = StockPriceForecaster(ticker=ticker.upper(), sequence_length=sequence_length,
                                             lstm_units=lstm_units, dropout=dropout)

            with st.spinner("Downloading stock data..."):
                forecaster.download_data(start_date=start_date, end_date=end_date)

            if use_indicators:
                with st.spinner("Calculating technical indicators..."):
                    forecaster.calculate_technical_indicators()

            with st.spinner("Preparing training sequences..."):
                X_train, y_train, X_test, y_test = forecaster.prepare_sequences()

            with st.spinner("Building and training LSTM model..."):
                forecaster.build_model(num_layers=num_layers)
                history = forecaster.train(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1)

            with st.spinner("Evaluating model performance..."):
                metrics, predictions, actual = forecaster.evaluate(X_test, y_test)

            st.session_state.forecaster = forecaster
            st.session_state.metrics = metrics
            st.session_state.predictions = predictions
            st.session_state.actual = actual
            st.session_state.trained = True

            status_placeholder.success("✅ Model trained successfully!")

        except Exception as e:
            status_placeholder.error(f"❌ Error: {str(e)}")
            st.session_state.trained = False

    # Display results if model is trained
    if st.session_state.trained:
        forecaster = st.session_state.forecaster
        metrics = st.session_state.metrics
        predictions = st.session_state.predictions
        actual = st.session_state.actual

        # Display metrics
        with metrics_placeholder.container():
            st.metric("MAE", f"${metrics['MAE']:.2f}")
            st.metric("RMSE", f"${metrics['RMSE']:.2f}")
            st.metric("MAPE", f"{metrics['MAPE']:.2f}%")

        # Generate future forecast
        with st.spinner("Generating forecast..."):
            future_prices = forecaster.forecast_future(days=forecast_days)

        # Create visualization
        with chart_placeholder.container():
            historical_data = forecaster.data.tail(200)
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'],
                                    mode='lines', name='Historical', line=dict(color='blue', width=2)))

            test_dates = historical_data.index[-len(predictions):]
            fig.add_trace(go.Scatter(x=test_dates, y=predictions.flatten(),
                                    mode='lines', name='Predictions', line=dict(color='orange', width=2, dash='dash')))

            last_date = historical_data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
            fig.add_trace(go.Scatter(x=future_dates, y=future_prices,
                                    mode='lines', name='Forecast', line=dict(color='green', width=2, dash='dot')))

            fig.update_layout(title=f"{ticker.upper()} Stock Price", xaxis_title="Date",
                            yaxis_title="Price ($)", height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Forecast summary
        st.subheader("Forecast Summary")
        col1, col2, col3 = st.columns(3)
        current_price = historical_data['Close'].iloc[-1]
        forecast_price = future_prices[-1]
        price_change = forecast_price - current_price
        percent_change = (price_change / current_price) * 100

        col1.metric("Current Price", f"${current_price:.2f}")
        col2.metric(f"Forecast ({forecast_days} days)", f"${forecast_price:.2f}",
                   f"{price_change:+.2f} ({percent_change:+.2f}%)")
        col3.metric("Trend", "📈" if percent_change > 0 else "📉")

    else:
        status_placeholder.info("👈 Configure parameters and click 'Train Model' to get started.")


# ============================================================================
# TAB 2: PORTFOLIO FORECAST
# ============================================================================
with tab2:
    st.header("💼 Portfolio Forecasting")
    st.markdown("Forecast your entire portfolio value 365 days into the future with risk metrics.")

    # Portfolio input section
    st.subheader("1️⃣ Enter Your Portfolio")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Portfolio input methods
        input_method = st.radio("Input Method:", ["Manual Entry", "Upload CSV"], horizontal=True)

        if input_method == "Manual Entry":
            st.markdown("**Enter your holdings (one per line in format: TICKER,QUANTITY)**")
            portfolio_text = st.text_area(
                "Portfolio Holdings",
                value="AAPL,100\nMSFT,50\nGOOGL,25\nAMZN,20\nNVDA,75",
                height=150,
                help="Example: AAPL,100 means 100 shares of Apple"
            )

            # Parse portfolio
            portfolio_holdings = {}
            if portfolio_text:
                for line in portfolio_text.strip().split('\n'):
                    if ',' in line:
                        parts = line.strip().split(',')
                        if len(parts) == 2:
                            ticker_sym = parts[0].strip().upper()
                            try:
                                quantity = float(parts[1].strip())
                                portfolio_holdings[ticker_sym] = quantity
                            except ValueError:
                                st.warning(f"Invalid quantity for {ticker_sym}")

        else:  # Upload CSV
            uploaded_file = st.file_uploader("Upload Portfolio CSV", type=['csv'])
            portfolio_holdings = {}

            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.markdown("**Uploaded Portfolio:**")
                st.dataframe(df, use_container_width=True)

                # Try to find ticker and quantity columns
                if 'ticker' in df.columns.str.lower() and 'quantity' in df.columns.str.lower():
                    ticker_col = [c for c in df.columns if c.lower() == 'ticker'][0]
                    quantity_col = [c for c in df.columns if c.lower() == 'quantity'][0]
                    for _, row in df.iterrows():
                        portfolio_holdings[str(row[ticker_col]).upper()] = float(row[quantity_col])
                else:
                    st.error("CSV must have 'ticker' and 'quantity' columns")

    with col2:
        if portfolio_holdings:
            st.markdown("**Portfolio Summary**")
            st.metric("Total Positions", len(portfolio_holdings))
            st.metric("Total Shares", f"{sum(portfolio_holdings.values()):,.0f}")

    # Display parsed portfolio
    if portfolio_holdings:
        st.markdown("**Current Holdings:**")
        holdings_df = pd.DataFrame([
            {"Ticker": k, "Quantity": v} for k, v in portfolio_holdings.items()
        ])
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

    # Portfolio forecast settings
    st.subheader("2️⃣ Forecast Settings")

    col1, col2, col3 = st.columns(3)
    with col1:
        portfolio_forecast_days = st.selectbox("Forecast Period", [30, 90, 180, 365], index=3)
    with col2:
        portfolio_epochs = st.slider("Training Epochs (Portfolio)", 10, 50, 20, 5)
    with col3:
        confidence_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)

    # Run portfolio forecast
    run_portfolio_forecast = st.button("🚀 Run Portfolio Forecast", type="primary", use_container_width=True)

    if run_portfolio_forecast and portfolio_holdings and HAS_MODEL:
        st.markdown("---")
        st.subheader("📊 Portfolio Forecast Results")

        progress_bar = st.progress(0)
        status_text = st.empty()

        portfolio_results = {}
        current_portfolio_value = 0

        # Train models for each stock
        total_stocks = len(portfolio_holdings)

        for idx, (ticker_sym, quantity) in enumerate(portfolio_holdings.items()):
            status_text.text(f"Processing {ticker_sym} ({idx+1}/{total_stocks})...")
            progress_bar.progress((idx + 1) / total_stocks)

            try:
                # Initialize and train forecaster
                forecaster = StockPriceForecaster(ticker=ticker_sym, sequence_length=60,
                                                 lstm_units=50, dropout=0.2)
                forecaster.download_data()
                forecaster.calculate_technical_indicators()
                X_train, y_train, X_test, y_test = forecaster.prepare_sequences()
                forecaster.build_model(num_layers=2)
                forecaster.train(X_train, y_train, epochs=portfolio_epochs, batch_size=32, validation_split=0.1)
                metrics, predictions, actual = forecaster.evaluate(X_test, y_test)

                # Generate forecast
                future_prices = forecaster.forecast_future(days=portfolio_forecast_days)
                current_price = forecaster.data['Close'].iloc[-1]

                # Calculate residuals for confidence intervals
                residuals = actual.flatten() - predictions.flatten()
                std_error = np.std(residuals)

                # Store results
                portfolio_results[ticker_sym] = {
                    'quantity': quantity,
                    'current_price': current_price,
                    'forecast_prices': future_prices,
                    'forecast_price': future_prices[-1],
                    'std_error': std_error,
                    'metrics': metrics,
                    'current_value': current_price * quantity,
                    'forecast_value': future_prices[-1] * quantity
                }

                current_portfolio_value += current_price * quantity

            except Exception as e:
                st.warning(f"Failed to process {ticker_sym}: {str(e)}")
                continue

        status_text.text("✅ Portfolio forecast complete!")
        progress_bar.empty()

        if portfolio_results:
            # Calculate portfolio metrics
            forecast_portfolio_value = sum(r['forecast_value'] for r in portfolio_results.values())
            portfolio_change = forecast_portfolio_value - current_portfolio_value
            portfolio_change_pct = (portfolio_change / current_portfolio_value) * 100

            # Calculate portfolio volatility
            z_score = {90: 1.645, 95: 1.96, 99: 2.576}[confidence_level]
            portfolio_std = np.sqrt(sum((r['std_error'] * r['quantity'])**2 for r in portfolio_results.values()))

            lower_bound = forecast_portfolio_value - z_score * portfolio_std
            upper_bound = forecast_portfolio_value + z_score * portfolio_std

            # Display key metrics
            st.markdown("### 🎯 Portfolio Value Projection")
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Current Value", f"${current_portfolio_value:,.2f}")
            col2.metric(f"Forecast ({portfolio_forecast_days}d)", f"${forecast_portfolio_value:,.2f}",
                       f"{portfolio_change:+,.2f} ({portfolio_change_pct:+.1f}%)")
            col3.metric(f"Lower Bound ({confidence_level}%)", f"${lower_bound:,.2f}")
            col4.metric(f"Upper Bound ({confidence_level}%)", f"${upper_bound:,.2f}")

            # Portfolio composition
            st.markdown("### 📈 Individual Stock Forecasts")
            results_df = pd.DataFrame([
                {
                    "Ticker": ticker_sym,
                    "Quantity": r['quantity'],
                    "Current Price": f"${r['current_price']:.2f}",
                    "Forecast Price": f"${r['forecast_price']:.2f}",
                    "Price Change %": f"{((r['forecast_price'] - r['current_price']) / r['current_price'] * 100):+.1f}%",
                    "Current Value": f"${r['current_value']:,.2f}",
                    "Forecast Value": f"${r['forecast_value']:,.2f}",
                    "Value Change": f"${(r['forecast_value'] - r['current_value']):+,.2f}",
                    "MAPE": f"{r['metrics']['MAPE']:.2f}%"
                }
                for ticker_sym, r in portfolio_results.items()
            ])
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Visualization: Portfolio value over time
            st.markdown("### 📊 Portfolio Value Trajectory")

            fig_portfolio = go.Figure()

            # Calculate portfolio value for each day
            dates = pd.date_range(start=datetime.now(), periods=portfolio_forecast_days + 1)
            portfolio_values = [current_portfolio_value]

            for day in range(portfolio_forecast_days):
                daily_value = sum(
                    r['forecast_prices'][day] * r['quantity']
                    for r in portfolio_results.values()
                )
                portfolio_values.append(daily_value)

            fig_portfolio.add_trace(go.Scatter(
                x=dates, y=portfolio_values,
                mode='lines', name='Portfolio Value',
                line=dict(color='green', width=3)
            ))

            # Add confidence bands
            ci_upper = [current_portfolio_value + (upper_bound - forecast_portfolio_value) * i / portfolio_forecast_days
                       for i in range(portfolio_forecast_days + 1)]
            ci_lower = [current_portfolio_value + (lower_bound - forecast_portfolio_value) * i / portfolio_forecast_days
                       for i in range(portfolio_forecast_days + 1)]

            fig_portfolio.add_trace(go.Scatter(
                x=dates, y=ci_upper,
                mode='lines', name=f'{confidence_level}% Upper',
                line=dict(color='lightgreen', width=1, dash='dash')
            ))
            fig_portfolio.add_trace(go.Scatter(
                x=dates, y=ci_lower,
                mode='lines', name=f'{confidence_level}% Lower',
                line=dict(color='lightgreen', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(144, 238, 144, 0.2)'
            ))

            fig_portfolio.update_layout(
                title=f"Portfolio Value Forecast ({portfolio_forecast_days} Days)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig_portfolio, use_container_width=True)

            # Risk metrics
            st.markdown("### ⚠️ Risk Metrics")
            col1, col2, col3, col4 = st.columns(4)

            # Portfolio volatility (annualized)
            daily_vol = portfolio_std / np.sqrt(portfolio_forecast_days)
            annual_vol = daily_vol * np.sqrt(252)
            annual_vol_pct = (annual_vol / current_portfolio_value) * 100

            # Value at Risk
            var_amount = z_score * daily_vol * current_portfolio_value

            # Expected return
            expected_return_annual = (portfolio_change / current_portfolio_value) * (365 / portfolio_forecast_days) * 100

            # Sharpe-like ratio (assuming 4% risk-free rate)
            risk_free_rate = 4.0
            sharpe_ratio = (expected_return_annual - risk_free_rate) / annual_vol_pct if annual_vol_pct > 0 else 0

            col1.metric("Portfolio Volatility", f"{annual_vol_pct:.2f}%", "Annualized")
            col2.metric(f"VaR ({confidence_level}%)", f"${var_amount:,.2f}", "Daily")
            col3.metric("Expected Return", f"{expected_return_annual:+.2f}%", "Annualized")
            col4.metric("Risk-Adj. Return", f"{sharpe_ratio:.2f}", "Sharpe-like")

            # Download results
            st.markdown("### 💾 Export Results")
            export_data = []
            for date, value in zip(dates, portfolio_values):
                export_data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Portfolio_Value': value
                })

            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Portfolio Forecast CSV",
                data=csv,
                file_name=f"portfolio_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This tool is for educational purposes. Predictions should not be used as sole investment advice.
Always consult with financial professionals before making investment decisions.
""")