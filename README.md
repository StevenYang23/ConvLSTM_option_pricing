# ConvLSTM Option Price Prediction Model

## Overview

This project implements a Convolutional LSTM (ConvLSTM) model for predicting option prices using historical data from the Polygon API. The model combines convolutional layers for spatial feature extraction (e.g., stock price, time-to-maturity, strike price, implied volatility) with LSTM layers for temporal modeling of option price sequences.

The notebook (`LSTM.ipynb`) demonstrates:
- Data fetching and preparation for a specific AAPL call option (symbol: `AAPL251219C00240000`, expiration: December 19, 2025, strike: $240).
- Implied volatility calculation using the Black-Scholes model and Brent's method for root finding.
- Model architecture, training, and evaluation.
- Visualization of stock prices, actual vs. predicted option prices, and data validation.

Key features:
- Uses PyTorch for the neural network.
- Handles sequence data with a sliding window approach (sequence length: 15 days).
- Trains on 90% of the data and tests on the remaining 10%.
- Focuses on predicting option prices for the test period (e.g., August 20, 2025, to September 24, 2025, based on sample output).

## Requirements

- Python 3.12+
- Libraries (install via `pip install -r requirements.txt`):
  - torch
  - numpy
  - scipy
  - pandas
  - matplotlib
  - polygon-api-client (for data fetching)
  - scikit-learn

Create a `requirements.txt` file with:
