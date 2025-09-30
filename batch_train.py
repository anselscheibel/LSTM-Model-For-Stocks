"""
Portfolio-Specific Batch Training Script

Trains LSTM models for all stocks in your portfolio with optimized settings
for long-term (365-day) forecasting.

Author: Ansel Scheibel
Date: December 2024
"""

import pandas as pd
from lstm_stock_forecaster import StockPriceForecaster
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# ============================================================================
# YOUR PORTFOLIO - REPLACE THIS WITH YOUR ACTUAL HOLDINGS
# ============================================================================
PORTFOLIO = {
    # Format: 'TICKER': quantity
    'AAPL': 100,
    'MSFT': 50,
    'GOOGL': 25,
    'AMZN': 20,
    'NVDA': 75,
    'TSLA': 30,
    'META': 40,
    'V': 45,
    'JPM': 60,
    'WMT': 55
}

# Alternative: Load from CSV
# portfolio_df = pd.read_csv('my_portfolio.csv')
# PORTFOLIO = dict(zip(portfolio_df['ticker'], portfolio_df['quantity']))

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
CONFIG = {
    'sequence_length': 60,      # Days of history to use
    'lstm_units': 50,            # LSTM layer size
    'num_layers': 2,             # Number of LSTM layers
    'dropout': 0.2,              # Dropout rate
    'epochs': 30,                # Training epochs (increase for better accuracy)
    'batch_size': 32,
    'validation_split': 0.1
}

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def train_portfolio_models():
    """Train LSTM models for all stocks in portfolio."""

    print("=" * 80)
    print("PORTFOLIO BATCH TRAINING")
    print("=" * 80)
    print(f"\nPortfolio Size: {len(PORTFOLIO)} positions")
    print(f"Total Shares: {sum(PORTFOLIO.values()):,.0f}")
    print(f"\nHoldings:")
    for ticker, qty in PORTFOLIO.items():
        print(f"  {ticker}: {qty:,.0f} shares")
    print("\n" + "=" * 80)

    results = []
    successful_models = []
    failed_models = []

    # Create directories
    Path('models/portfolio').mkdir(parents=True, exist_ok=True)
    Path('outputs/portfolio').mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    for idx, (ticker, quantity) in enumerate(PORTFOLIO.items(), 1):
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(PORTFOLIO)}] Training {ticker} ({quantity:,.0f} shares)")
        print(f"{'=' * 80}")

        try:
            # Initialize forecaster
            forecaster = StockPriceForecaster(
                ticker=ticker,
                sequence_length=CONFIG['sequence_length'],
                lstm_units=CONFIG['lstm_units'],
                dropout=CONFIG['dropout']
            )

            # Download data
            print(f"\n📥 Downloading {ticker} data...")
            forecaster.download_data()
            current_price = forecaster.data['Close'].iloc[-1]
            current_value = current_price * quantity

            # Calculate indicators
            print(f"🔧 Calculating technical indicators...")
            forecaster.calculate_technical_indicators()

            # Prepare sequences
            print(f"🔄 Preparing sequences...")
            X_train, y_train, X_test, y_test = forecaster.prepare_sequences()

            # Build and train model
            print(f"🏗️  Building model...")
            forecaster.build_model(num_layers=CONFIG['num_layers'])

            print(f"🎯 Training model ({CONFIG['epochs']} epochs)...")
            history = forecaster.train(
                X_train, y_train,
                epochs=CONFIG['epochs'],
                batch_size=CONFIG['batch_size'],
                validation_split=CONFIG['validation_split']
            )

            # Evaluate
            print(f"📊 Evaluating...")
            metrics, predictions, actual = forecaster.evaluate(X_test, y_test)

            # Generate 365-day forecast
            print(f"🔮 Generating 365-day forecast...")
            forecast_365 = forecaster.forecast_future(days=365)
            forecast_price = forecast_365[-1]
            forecast_value = forecast_price * quantity

            # Calculate confidence metrics
            residuals = actual.flatten() - predictions.flatten()
            std_error = np.std(residuals)

            # 95% confidence interval for 365-day forecast
            ci_95_lower = forecast_price - 1.96 * std_error
            ci_95_upper = forecast_price + 1.96 * std_error

            # Store results
            result = {
                'ticker': ticker,
                'quantity': quantity,
                'current_price': float(current_price),
                'current_value': float(current_value),
                'forecast_price_365d': float(forecast_price),
                'forecast_value_365d': float(forecast_value),
                'value_change': float(forecast_value - current_value),
                'value_change_pct': float((forecast_value - current_value) / current_value * 100),
                'price_change_pct': float((forecast_price - current_price) / current_price * 100),
                'ci_95_lower': float(ci_95_lower),
                'ci_95_upper': float(ci_95_upper),
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE'],
                'std_error': float(std_error),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }

            results.append(result)
            successful_models.append(ticker)

            # Save model
            print(f"💾 Saving model...")
            forecaster.save_model(path='models/portfolio')

            # Save individual forecast
            forecast_df = pd.DataFrame({
                'day': range(1, 366),
                'date': pd.date_range(start=datetime.now(), periods=365),
                'predicted_price': forecast_365
            })
            forecast_df.to_csv(f'outputs/portfolio/{ticker}_365day_forecast.csv', index=False)

            print(f"\n✅ {ticker} completed successfully!")
            print(f"   Current: ${current_price:.2f} → Forecast: ${forecast_price:.2f} ({result['price_change_pct']:+.1f}%)")
            print(f"   Position value: ${current_value:,.2f} → ${forecast_value:,.2f} ({result['value_change_pct']:+.1f}%)")

        except Exception as e:
            print(f"\n❌ Failed on {ticker}: {str(e)}")
            failed_models.append({'ticker': ticker, 'error': str(e)})
            continue

    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds() / 60

    # ============================================================================
    # PORTFOLIO SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("PORTFOLIO SUMMARY")
    print("=" * 80)

    if results:
        # Calculate portfolio totals
        current_portfolio_value = sum(r['current_value'] for r in results)
        forecast_portfolio_value = sum(r['forecast_value_365d'] for r in results)
        portfolio_change = forecast_portfolio_value - current_portfolio_value
        portfolio_change_pct = (portfolio_change / current_portfolio_value) * 100

        # Calculate portfolio-wide confidence interval
        portfolio_std = np.sqrt(sum((r['std_error'] * r['quantity'])**2 for r in results))
        portfolio_ci_lower = forecast_portfolio_value - 1.96 * portfolio_std
        portfolio_ci_upper = forecast_portfolio_value + 1.96 * portfolio_std

        print(f"\n📊 Portfolio Value:")
        print(f"   Current:       ${current_portfolio_value:,.2f}")
        print(f"   Forecast (1Y): ${forecast_portfolio_value:,.2f}")
        print(f"   Change:        ${portfolio_change:+,.2f} ({portfolio_change_pct:+.2f}%)")
        print(f"\n📈 95% Confidence Interval:")
        print(f"   Lower Bound:   ${portfolio_ci_lower:,.2f}")
        print(f"   Upper Bound:   ${portfolio_ci_upper:,.2f}")

        # Top gainers and losers
        sorted_by_change = sorted(results, key=lambda x: x['value_change_pct'], reverse=True)

        print(f"\n🏆 Top 3 Expected Gainers:")
        for r in sorted_by_change[:3]:
            print(f"   {r['ticker']}: {r['value_change_pct']:+.1f}% (${r['value_change']:+,.2f})")

        print(f"\n⚠️  Top 3 Expected Decliners:")
        for r in sorted_by_change[-3:]:
            print(f"   {r['ticker']}: {r['value_change_pct']:+.1f}% (${r['value_change']:+,.2f})")

        # Model performance
        avg_mape = sum(r['MAPE'] for r in results) / len(results)
        avg_mae = sum(r['MAE'] for r in results) / len(results)

        print(f"\n🎯 Average Model Performance:")
        print(f"   MAPE: {avg_mape:.2f}%")
        print(f"   MAE:  ${avg_mae:.2f}")

        # Save comprehensive results
        results_df = pd.DataFrame(results)
        results_df.to_csv('outputs/portfolio/portfolio_forecast_summary.csv', index=False)

        # Save JSON with metadata
        summary = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_size': len(PORTFOLIO),
            'successful_models': len(successful_models),
            'failed_models': len(failed_models),
            'training_duration_minutes': round(training_duration, 2),
            'current_portfolio_value': float(current_portfolio_value),
            'forecast_portfolio_value': float(forecast_portfolio_value),
            'portfolio_change': float(portfolio_change),
            'portfolio_change_pct': float(portfolio_change_pct),
            'ci_95_lower': float(portfolio_ci_lower),
            'ci_95_upper': float(portfolio_ci_upper),
            'average_mape': float(avg_mape),
            'config': CONFIG,
            'results': results,
            'failed': failed_models
        }

        with open('outputs/portfolio/portfolio_forecast_complete.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Results saved to outputs/portfolio/")

    print(f"\n✅ Training completed in {training_duration:.1f} minutes")
    print(f"   Successful: {len(successful_models)}/{len(PORTFOLIO)}")
    if failed_models:
        print(f"   Failed: {len(failed_models)} - {[f['ticker'] for f in failed_models]}")

    print("\n" + "=" * 80)

    return results


# ============================================================================
# QUICK PORTFOLIO VALUE CALCULATOR
# ============================================================================
def calculate_current_portfolio_value():
    """Quick calculation of current portfolio value without training."""
    import yfinance as yf

    print("\n📊 Calculating current portfolio value...")
    total_value = 0

    for ticker, quantity in PORTFOLIO.items():
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            value = current_price * quantity
            total_value += value
            print(f"   {ticker}: {quantity:,.0f} shares × ${current_price:.2f} = ${value:,.2f}")
        except Exception as e:
            print(f"   {ticker}: Error - {e}")

    print(f"\n   Total Portfolio Value: ${total_value:,.2f}")
    return total_value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train LSTM models for portfolio')
    parser.add_argument('--quick-value', action='store_true',
                       help='Just calculate current portfolio value without training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')

    args = parser.parse_args()

    if args.quick_value:
        calculate_current_portfolio_value()
    else:
        if args.epochs:
            CONFIG['epochs'] = args.epochs
        train_portfolio_models()