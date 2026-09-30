# ⚡ Electric Vehicle (EV) Charging & Renewable Energy Forecasting

[![Open QLSTM In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sainathkotage/ev-charging-forecasting/blob/main/quantum_lstm_renewable_ev_forecasting.ipynb)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.40%2B-purple.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)

> A collection of production-grade Machine Learning and Quantum Machine Learning (QML) forecasting systems designed for **Google Colab**. This repository features:
> 1. **Quantum LSTM (QLSTM)**: Forecasting 1-hour ahead renewable power generation ($kW$) with PennyLane + PyTorch and estimating EV fleet charging capacity ($60\text{ kWh}$ battery capacity).
> 2. **Classical Ensemble & Deep Learning**: End-to-end hourly EV charging load demand forecasting with solar PV co-generation and peak-shaving simulation.

---

## 📌 Available Notebooks

| Notebook / App | Focus | Key Models | Access / Launch |
| :--- | :--- | :--- | :---: |
| [`app.py`](app.py) & [`dashboard.html`](dashboard.html) | **⚡ Interactive EV Fleet Charging & Dispatch Dashboard** | Real-time QLSTM inference, What-if sliders, KPI cards | `python -m streamlit run app.py` or open `dashboard.html` |
| [`quantum_lstm_renewable_ev_forecasting.ipynb`](quantum_lstm_renewable_ev_forecasting.ipynb) | **1-Hour Ahead Renewable Generation & EV Fleet Charging** | Variational Quantum LSTM (PennyLane), Classical LSTM, Persistence Baseline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sainathkotage/ev-charging-forecasting/blob/main/quantum_lstm_renewable_ev_forecasting.ipynb) |
| [`ev_charging_forecasting.ipynb`](ev_charging_forecasting.ipynb) | **Hourly EV Charging Load Demand Forecasting** | HistGradientBoosting, Random Forest, Ridge, MLP Regressor | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sainathkotage/ev-charging-forecasting/blob/main/ev_charging_forecasting.ipynb) |

---

## ⚛️ Quantum LSTM (QLSTM) Architecture & Workflow

The QLSTM model predicts renewable generation $1\text{ hour ahead}$ using:
- **Angle Embedding**: Encodes classical feature projections into single-qubit rotation angles around the Y-axis ($R_y$).
- **Ansatz (`StronglyEntanglingLayers`)**: Parameterized unitary rotations and entangling CNOT gates across 4 qubits.
- **Expectation Measurements**: Evaluates Pauli-Z expectation values ($\langle Z \rangle \in [-1, 1]$) across all 4 LSTM gates (Forget, Input, Candidate, Output).
- **Leakage Prevention**: Chronological 70% / 15% / 15% split; `MinMaxScaler` is fit strictly on training data.
- **EV Charging Calculation**: Converts predicted hourly power ($kW$) to energy ($kWh$) and calculates exact and integer (whole) EVs fully chargeable assuming $60\text{ kWh}$ battery capacity.

```
                      [ Environmental Signals & Lags ]
                                      |
                                      v
                       +-------------------------------+
                       |  Feature Projection to Qubits |
                       +-------------------------------+
                                      |
                    +-----------------+-----------------+
                    |                 |                 |
                    v                 v                 v
             [ Angle Embedding ] [ Entangling Layers ] [ Pauli-Z Measurement ]
                    |                 |                 |
                    +-----------------+-----------------+
                                      |
                                      v
                       +-------------------------------+
                       |  Quantum Gated LSTM Cell (x4) |
                       |    Forget, Input, Cell, Out   |
                       +-------------------------------+
                                      |
                                      v
                       +-------------------------------+
                       |  Linear Output -> kW at t+1   |
                       +-------------------------------+
                                      |
                                      v
                       +-------------------------------+
                       | EV Fleet Charging Estimation  |
                       |   EVs = Total Energy / 60 kWh |
                       +-------------------------------+
```

---

## 📊 Benchmark Results

### 1. Renewable Generation Forecasting (1-Hour Ahead, R² Target: 0.96 - 0.98)
Evaluated on an out-of-sample chronological test set with 1-hour ahead Numerical Weather Predictions (NWP):

| Model | RMSE ($kW$) ↓ | MAE ($kW$) ↓ | $R^2$ Score ↑ | sMAPE (%) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **Quantum-Enhanced LSTM (QLSTM)** | **7.87** | **5.29** | **0.9745** | **67.63%** |
| **Classical Baseline** | 7.73 | 5.03 | 0.9754 | 64.76% |
| **Persistence Baseline ($t-1$)** | 34.26 | 20.22 | 0.5166 | 98.01% |

### 2. EV Fleet Charging Estimation (60 kWh Battery)
- **Total Actual Generation**: $27,182.18\text{ kWh}$ $\to$ **$453.04$ EVs** ($453$ whole EVs)
- **Total Quantum Forecast**: $28,438.49\text{ kWh}$ $\to$ **$473.97$ EVs** ($473$ whole EVs)
- **Forecasting Deviation**: **$+4.62\%$** error margin ($+20$ whole EVs over the test period).
- Provides critical lookahead for distribution system operators to schedule EV smart charging during solar surplus hours.

---

## 📂 Repository Structure

```
├── .gitignore                                      # Standard Git ignore rules
├── LICENSE                                         # MIT License
├── README.md                                       # Documentation & Colab quick start
├── requirements.txt                                # Python, PennyLane & PyTorch dependencies
├── quantum_lstm_renewable_ev_forecasting.ipynb     # Quantum LSTM Notebook for Google Colab
├── ev_charging_forecasting.ipynb                   # Classical EV Demand Forecasting Notebook
├── ev_charging_forcasting.ipynb                    # Exact name match notebook
├── generate_data.py                                # Synthetic EV charging data generator
├── build_notebook.py                               # Builder script for EV load notebook
├── build_qlstm_notebook.py                         # Builder script for QLSTM notebook
├── reg_forecasting_v2_project_clean.csv            # Clean renewable energy dataset (4,368 records)
├── data/
│   ├── ev_charging_data.csv                        # 18 months of hourly EV load data
│   └── reg_forecasting_v2_project_clean.csv        # Mirror copy of renewable dataset
└── results/                                        # Generated plots, predictions CSV & metric JSON
    ├── qlstm_best_model.pt
    ├── predictions.csv
    ├── metrics_summary.json
    ├── loss_curve.png
    ├── test_actual_vs_predicted.png
    ├── test_7day_zoom.png
    ├── scatter_actual_vs_predicted.png
    └── ev_charging_analysis.png
```

---

## 🚀 Quick Start in Google Colab

1. **Open the Quantum LSTM Notebook:**
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sainathkotage/ev-charging-forecasting/blob/main/quantum_lstm_renewable_ev_forecasting.ipynb)

2. Click **Runtime** > **Run all** (`Ctrl + F9`).
   - The notebook automatically detects if running in Colab and installs `pennylane`.
   - If `reg_forecasting_v2_project_clean.csv` is uploaded or mounted from Drive, it uses it directly; otherwise it automatically generates the dataset with physical solar & wind equations so that the notebook **never fails**.
   - Trains the QLSTM in minutes and generates diagnostic plots in `results/`.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
