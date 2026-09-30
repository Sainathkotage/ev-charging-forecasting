"""
Script: generate_t1_ev_table.py
Generates the exact 1-hour ahead (t+1) renewable energy prediction table,
error metrics summary block, and Canva/presentation slide graphics
matching the user's requested slide template.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# 1. Load predictions
pred_path = "results/predictions.csv"
if not os.path.exists(pred_path):
    raise FileNotFoundError("results/predictions.csv not found. Run run_upgrade_pipeline.py first.")

df = pd.read_csv(pred_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

act = df['actual_generation_kw'].values
pred = df['qlstm_forecast_kw'].values
timestamps = df['timestamp']

# 2. Build the exact table structure requested by the user
# Columns: datetime, Actual_Renewable, QNN_Predicted_Renewable, Absolute_Error, Percentage_Error, Num_EVs_Charged
abs_err = np.abs(act - pred)
# Safe percentage error: if actual > 0, (abs_err / act)*100, else 0.0
pct_err = np.zeros_like(act)
pos_act_mask = act > 1e-4
pct_err[pos_act_mask] = (abs_err[pos_act_mask] / act[pos_act_mask]) * 100.0
num_evs = pred / 60.0

table_df = pd.DataFrame({
    'datetime': timestamps.dt.strftime('%Y-%m-%d %H:%M:%S'),
    'Actual_Renewable': np.round(act, 1),
    'QNN_Predicted_Renewable': np.round(pred, 6),
    'Absolute_Error': np.round(abs_err, 6),
    'Percentage_Error': np.round(pct_err, 6),
    'Num_EVs_Charged': np.round(num_evs, 6)
})

# Save table to CSV
table_csv_path = "results/t1_ev_charging_table.csv"
table_df.to_csv(table_csv_path, index=False)
print(f"[OK] Saved {table_csv_path}")

# 3. Compute Metrics
mae = float(np.mean(abs_err))
rmse = float(np.sqrt(np.mean(abs_err**2)))
# MAPE on positive actuals (> 1 kW to prevent unbounded percentage on 0.01 kW nighttime solar)
pos_mask = act > 1.0
mape = float(np.mean(np.abs(act[pos_mask] - pred[pos_mask]) / act[pos_mask]) * 100.0) if np.sum(pos_mask) > 0 else 0.0
smape = float(np.mean(2.0 * abs_err / (np.abs(act) + np.abs(pred) + 1e-6)) * 100.0)
mean_act = float(np.mean(act))
norm_mae = float((mae / mean_act) * 100.0)
norm_rmse = float((rmse / mean_act) * 100.0)
ss_res = np.sum((act - pred)**2)
ss_tot = np.sum((act - mean_act)**2)
r2 = float(1.0 - (ss_res / ss_tot))

avg_evs = float(np.mean(num_evs))

print("\n" + "=" * 50)
print("1-HOUR AHEAD QNN / QLSTM RESULTS (t -> t+1)")
print("=" * 50)
print(f"MAE                  : {mae:.4f}")
print(f"MAPE                 : {mape:.4f}%")
print(f"SMAPE                : {smape:.4f}%")
print(f"Normalized MAE       : {norm_mae:.4f}%")
print(f"Normalized RMSE      : {norm_rmse:.4f}%")
print(f"R2                   : {r2:.6f}")
print("=" * 50)
print(f"Average number of EVs that can be charged per hour: {avg_evs:.12f}")
print("=" * 50)

print("\nFirst 10 Rows of t+1 Prediction Table:")
print(table_df.head(10).to_string())

# 4. Generate Presentation Slide Graphic 1: Green EVs Chargeable per Hour Line Plot
plt.figure(figsize=(12, 5.5))
plt.plot(timestamps, num_evs, color='green', linewidth=1.2, label='Number of EVs Charged')
plt.title('Predicted Number of EVs Chargeable per Hour\nNumber of EVs that can be Charged (60 kWh battery)', fontsize=13, fontweight='bold')
plt.xlabel('Timestamp', fontsize=11)
plt.ylabel('Number of EVs', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig("results/slide1_predicted_evs_hourly.png", dpi=300)
plt.close()
print("[OK] Saved results/slide1_predicted_evs_hourly.png")

# 5. Generate Presentation Slide Graphic 2: Actual vs Predicted Scatter Plot
plt.figure(figsize=(6.5, 6.5))
plt.scatter(act, pred, alpha=0.45, color='#1f77b4', s=20, edgecolors='none')
max_val = max(act.max(), pred.max()) * 1.05
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y = x')
plt.title(f'Actual vs Predicted\nR² = {r2:.4f}', fontsize=13, fontweight='bold')
plt.xlabel('Actual Renewable Energy (kW)', fontsize=11)
plt.ylabel('Predicted Renewable Energy (kW)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig("results/slide2_actual_vs_predicted_scatter.png", dpi=300)
plt.close()
print("[OK] Saved results/slide2_actual_vs_predicted_scatter.png")

# 6. Save formatted summary JSON
summary = {
    "model": "1-Hour Ahead QNN / QLSTM",
    "forecast_horizon": "1 hour ahead (t+1)",
    "metrics": {
        "MAE": round(mae, 4),
        "MAPE_pct": round(mape, 4),
        "SMAPE_pct": round(smape, 4),
        "Normalized_MAE_pct": round(norm_mae, 4),
        "Normalized_RMSE_pct": round(norm_rmse, 4),
        "R2": round(r2, 6)
    },
    "average_evs_chargeable_per_hour": round(avg_evs, 6),
    "sample_table_head": table_df.head(10).to_dict(orient="records")
}

with open("results/t1_ev_summary.json", "w") as f:
    json.dump(summary, f, indent=4)
print("[OK] Saved results/t1_ev_summary.json")
