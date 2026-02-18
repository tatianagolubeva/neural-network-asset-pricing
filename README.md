# Neural Network Asset Pricing: Echo State Network vs. Linear Regression

Comparison of an Echo State Network (ESN) and a baseline linear regression model for short-term stock price forecasting, applied to Apple (AAPL) historical daily closing prices.

**Course:** Advanced Methods of Risk Management — University of Bologna, MSc Quantitative Finance (2024)

---

## Overview

Traditional asset pricing models rely on assumptions that often don't hold in real-world data. This project explores whether a recurrent neural network architecture (ESN) can outperform a simple linear baseline for next-day closing price prediction, and under what conditions complexity adds value — drawing on the "virtue of complexity" framework from Didisheim et al. (2023).

---

## Methods

| Model | Description |
|---|---|
| **Echo State Network (ESN)** | Reservoir computing model with sparsely connected recurrent hidden layer; output trained via Ridge regression |
| **Linear Regression** | Autoregressive baseline using previous *n* days as features |

**Approach:**
- Full AAPL historical daily close prices fetched via Alpha Vantage API
- Min-Max scaling applied as preprocessing
- 30-day lookback window to predict next-day close
- 80/20 train/test split (no shuffle — temporal order preserved)
- ESN hyperparameters tuned via `GridSearchCV` (polynomial degree, Ridge alpha)

---

## Results

| Model | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| ESN (baseline) | 1.97e-05 | 4.44e-03 | 3.57e-03 | 0.9899 |
| Linear Regression | 1.28e-05 | 3.57e-03 | 1.68e-03 | 0.9934 |
| **ESN (tuned)** | **1.28e-05** | **3.58e-03** | **1.68e-03** | **0.9934** |

Both models achieve R² > 0.99. After hyperparameter tuning, the ESN matches the linear regression baseline — consistent with the literature finding that for near-linear financial time series, added model complexity does not always yield improvement. The similarity likely reflects a predominantly linear relationship in the input/output mapping, and potential ESN overfitting prior to tuning.

---

## Visualisation

Actual vs. predicted closing prices (scaled) for both models on the test set:

> *Plots available in `/plots` folder*

---

## Stack

- Python 3.x
- `scikit-learn` — ESN pipeline (PolynomialFeatures + Ridge), LinearRegression, GridSearchCV, metrics
- `numpy`, `pandas` — data handling
- `matplotlib` — visualisation
- `alpha_vantage` — historical price data
- `Git`

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/neural-network-asset-pricing.git
cd neural-network-asset-pricing
pip install -r requirements.txt
```

Add your Alpha Vantage API key (free at [alphavantage.co](https://www.alphavantage.co)):

```python
api_key = 'YOUR_API_KEY'
```

Then run:

```bash
jupyter notebook esn_stock_prediction.ipynb
```

---

## References

1. Didisheim, A., Ke, S., Kelly, B., Malamud, S. (2023). *Complexity in Factor Pricing Models*
2. Lin, X., Yang, Z., Song, Y. (2009). *Short-term stock price prediction based on Echo State Networks*
3. Liu, S., Borovykh, A., Grzelak, L., Oosterlee, C. (2019). *A neural network-based framework for financial model calibration*
4. Kim, K. (2003). *Financial time series forecasting using Support Vector Machines*
5. Alhnaity, B., Abbod, M. (2020). *A new hybrid financial time series prediction model*

---

## Author

**Tatiana Golubeva** — MSc Quantitative Finance, University of Bologna  
[LinkedIn](https://linkedin.com/in/golubevatatiana)
