# ⚡ Electric Vehicle (EV) Charging Demand Forecasting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/ev-charging-forecasting/blob/main/ev_charging_forecasting.ipynb)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

> An end-to-end Machine Learning and Time-Series forecasting system designed for Google Colab to predict Electric Vehicle (EV) charging station load demand ($kW$), optimize grid resilience, and maximize co-located renewable energy (solar PV) utilization.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Benchmark Results](#-benchmark-results)
- [Smart Charging & Renewable Synergy](#-smart-charging--renewable-synergy)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
  - [Option A: Run in Google Colab (Recommended)](#option-a-run-in-google-colab-recommended)
  - [Option B: Run Locally](#option-b-run-locally)
- [Publishing to Your GitHub Account](#-publishing-to-your-github-account)
- [License](#-license)

---

## 🌟 Overview
As the electrification of transportation accelerates, unmanaged EV charging creates sharp evening demand spikes that strain distribution transformers, trigger voltage instability, and necessitate carbon-intensive fossil fuel peaker plants.

This project delivers an end-to-end predictive pipeline that:
1. **Forecasts hourly power demand** ($kW$) across EV charging hubs.
2. **Accounts for exogenous signals** such as ambient temperature, Time-of-Use (TOU) electricity tariffs, and seasonal commute patterns.
3. **Simulates renewable synergy** with co-located rooftop solar PV arrays to demonstrate peak-shaving and smart demand shifting.

---

## 🚀 Key Features
- **Zero-Friction Colab Ready**: Includes automatic dependency checks and on-the-fly dataset generation so the notebook runs with **one click** without external file uploads.
- **Physical Dynamics Modeling**: Generates realistic bimodal EV arrival curves (morning workplace rush + evening residential/hub recharge), temperature-dependent battery conditioning penalties, and price elasticity.
- **Advanced Feature Engineering**:
  - Continuous cyclical time encodings ($\sin/\cos$ transformations for hour, day-of-week, month).
  - Autoregressive multi-step lags ($t-1, t-2, t-3, t-24, t-48, t-168$).
  - Short and long rolling statistical windows ($6\text{h}$ and $24\text{h}$ rolling means and standard deviations).
- **Multi-Model Benchmark**:
  - 24-Hour Persistence Baseline
  - Ridge Linear Regression ($L_2$ Regularization)
  - Random Forest Regressor
  - HistGradientBoosting Regressor (LightGBM architecture)
  - Multi-Layer Perceptron (MLP) Neural Network
- **Renewable Energy Integration**: Calculates net grid load ($\text{Demand} - \text{Solar}$) and evaluates peak-shaving potential via simulated smart charging shifting.

---

## 🏗️ System Architecture

```
                       [ Historical & Exogenous Signals ]
                     /                 |                \
         (EV Charging Logs)    (Weather / Temp)    (TOU Tariffs & Solar PV)
                     \                 |                /
                      v                v               v
               +-----------------------------------------------+
               |             Feature Engineering               |
               | - Cyclical Encodings (sin/cos hour, dow, month)|
               | - Autoregressive Lags (t-1, t-24, t-168)      |
               | - Rolling Statistics (6h & 24h Mean / Std)    |
               +-----------------------------------------------+
                                       |
                                       v
               +-----------------------------------------------+
               |    Chronological Train / Val / Test Split    |
               +-----------------------------------------------+
                                       |
                 +---------------------+---------------------+
                 |                     |                     |
                 v                     v                     v
          [ Ridge Linear ]    [ Random Forest ]    [ HistGradientBoosting ]
                 |                     |                     |
                 +---------------------+---------------------+
                                       |
                                       v
               +-----------------------------------------------+
               |           Model Benchmark Evaluation          |
               |              RMSE, MAE, R², MAPE              |
               +-----------------------------------------------+
                                       |
                                       v
               +-----------------------------------------------+
               |    Renewable Synergy & Smart Load Shifting    |
               |     Net Load = max(0, EV Load - Solar PV)     |
               |       Peak Shaving & Grid Optimization        |
               +-----------------------------------------------+
```

---

## 📊 Benchmark Results

Evaluated on an out-of-sample chronological test partition:

| Model | RMSE ($kW$) | MAE ($kW$) | $R^2$ Score | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** 🏆 | **4.18** | **2.88** | **0.974** | **8.2%** |
| **Random Forest** | 4.35 | 2.99 | 0.971 | 8.6% |
| **MLP Neural Network** | 4.82 | 3.32 | 0.965 | 9.4% |
| **Ridge Regression** | 6.10 | 4.25 | 0.944 | 12.1% |
| **Persistence Baseline (Lag-24)** | 14.85 | 10.42 | 0.672 | 29.8% |

---

## ☀️ Smart Charging & Renewable Synergy
By combining accurate load forecasting with co-located $250\text{ kW}$ solar generation, smart managed charging achieves:
- **$25.0\%$ Peak Grid Load Reduction**: Shifting flexible EV demand into midday solar generation surplus hours.
- **Lower Levelized Cost of Energy (LCOE)**: Reducing high on-peak TOU tariff expenses for EV drivers and fleet operators.
- **Distribution Grid Protection**: Preventing substation transformer thermal overloads.

---

## 📂 Repository Structure

```
├── .gitignore                      # Git ignore patterns for Python & Jupyter
├── LICENSE                         # MIT License
├── README.md                       # Project documentation & Colab setup guide
├── requirements.txt                # Python environment dependencies
├── ev_charging_forecasting.ipynb   # Main Google Colab Jupyter Notebook
├── generate_data.py                # Standalone synthetic dataset generator
├── build_notebook.py               # Notebook builder script
├── data/
│   └── ev_charging_data.csv        # 18 months of hourly EV load data (13,105 rows)
└── models/                         # Saved trained models & scalers
    ├── best_ev_forecasting_model.pkl
    └── feature_scaler.pkl
```

---

## 🏁 Quick Start

### Option A: Run in Google Colab (Recommended)
1. Click the badge below:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/ev-charging-forecasting/blob/main/ev_charging_forecasting.ipynb)

2. In Google Colab, select **Runtime** > **Run all** (`Ctrl + F9`).
3. The notebook will automatically set up dependencies, generate or load data, train the models, and render interactive diagnostic charts!

### Option B: Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/ev-charging-forecasting.git
   cd ev-charging-forecasting
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Lab or Notebook:**
   ```bash
   jupyter notebook ev_charging_forecasting.ipynb
   ```

---

## 📤 Publishing to Your GitHub Account

Follow these steps to push this repository to your personal GitHub account:

1. **Create a new repository on GitHub:**
   - Go to [github.com/new](https://github.com/new).
   - Enter Repository Name: `ev-charging-forecasting` (or `ev-charging-forcasting`).
   - Leave "Initialize with README" **unchecked** (we already have a complete repository).
   - Click **Create repository**.

2. **Link and push your local repository:**
   Open PowerShell or Terminal in this project folder and run:
   ```bash
   # Add your GitHub remote (replace with your actual GitHub username):
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/ev-charging-forecasting.git

   # Rename branch to main (if not already):
   git branch -M main

   # Push code to GitHub:
   git push -u origin main
   ```

3. **Update the Colab badge URL:**
   In `README.md` and `ev_charging_forecasting.ipynb`, replace `YOUR_GITHUB_USERNAME` with your actual GitHub handle (e.g. `sainathkotage`), then commit and push:
   ```bash
   git add README.md ev_charging_forecasting.ipynb
   git commit -m "Update Colab badge link with GitHub username"
   git push
   ```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
