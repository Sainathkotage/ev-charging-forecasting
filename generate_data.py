"""
Synthetic Dataset Generator for Electric Vehicle (EV) Charging Load Forecasting
Generates realistic hourly EV charging demand records with weather, TOU pricing,
and co-located solar generation for grid integration analysis.
"""

import os
import numpy as np
import pandas as pd

def generate_ev_charging_dataset(
    start_date: str = "2024-01-01",
    end_date: str = "2025-06-30",
    freq: str = "1h",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic hourly EV charging station load data along with
    environmental, pricing, and co-located solar generation signals.
    """
    np.random.seed(random_seed)
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(timestamps)

    df = pd.DataFrame({"timestamp": timestamps})
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 1. Ambient Temperature (Celsius) with seasonal and diurnal cycles
    seasonal_temp = 15.0 + 10.0 * np.sin(2 * np.pi * (df["day_of_year"] - 105) / 365)
    diurnal_temp = 5.0 * np.sin(2 * np.pi * (df["hour"] - 9) / 24)
    temp_noise = np.random.normal(0, 2.0, n)
    df["temperature_c"] = np.round(seasonal_temp + diurnal_temp + temp_noise, 2)

    # 2. Time-of-Use (TOU) Electricity Price ($/kWh)
    def calculate_price(hour: int, is_wknd: int) -> float:
        if is_wknd:
            return 0.15 if (10 <= hour <= 18) else 0.10
        if 16 <= hour <= 21:
            return 0.38  # On-peak evening tariff
        elif (7 <= hour < 16) or (21 < hour <= 23):
            return 0.22  # Mid-peak commercial tariff
        else:
            return 0.12  # Off-peak overnight tariff

    df["electricity_price_kwh"] = [calculate_price(h, w) for h, w in zip(df["hour"], df["is_weekend"])]

    # 3. Co-located Solar PV Generation (kW) - 250 kW peak system
    solar_sun_elevation = np.maximum(0.0, np.sin(np.pi * (df["hour"] - 6) / 12)) ** 1.5
    solar_sun_elevation = np.where((df["hour"] >= 6) & (df["hour"] <= 18), solar_sun_elevation, 0.0)
    seasonal_solar_factor = 0.75 + 0.25 * np.sin(2 * np.pi * (df["day_of_year"] - 80) / 365)
    cloud_cover_factor = np.clip(np.random.normal(0.88, 0.12, n), 0.15, 1.0)
    df["solar_generation_kw"] = np.round(250.0 * solar_sun_elevation * seasonal_solar_factor * cloud_cover_factor, 2)

    # 4. EV Charging Demand (kW)
    # Weekday: Morning workplace rush (08:00-10:00) + Evening residential/hub recharge (17:00-21:00)
    # Weekend: Midday commercial/retail visit recharge (11:00-16:00)
    weekday_demand = (
        38.0 * np.exp(-((df["hour"] - 9.0) ** 2) / (2 * 1.8 ** 2)) +
        65.0 * np.exp(-((df["hour"] - 18.5) ** 2) / (2 * 2.4 ** 2)) +
        12.0 * np.exp(-((df["hour"] - 13.0) ** 2) / (2 * 2.0 ** 2)) +
        6.0
    )

    weekend_demand = (
        48.0 * np.exp(-((df["hour"] - 13.5) ** 2) / (2 * 3.2 ** 2)) +
        22.0 * np.exp(-((df["hour"] - 19.0) ** 2) / (2 * 2.2 ** 2)) +
        8.0
    )

    base_demand = np.where(df["is_weekend"] == 1, weekend_demand, weekday_demand)

    # Temperature adjustment: extreme cold (<5C) or hot (>30C) increases battery conditioning load
    temp_penalty = 1.0 + 0.015 * np.maximum(0.0, 5.0 - df["temperature_c"]) + 0.01 * np.maximum(0.0, df["temperature_c"] - 30.0)

    # Price sensitivity
    price_sensitivity = 1.0 - 0.20 * ((df["electricity_price_kwh"] - 0.12) / 0.26)

    # Stochastic arrival fluctuations
    noise = np.random.gamma(shape=4.0, scale=2.5, size=n) - 10.0

    raw_load = (base_demand * temp_penalty * price_sensitivity) + noise
    df["charging_load_kw"] = np.round(np.clip(raw_load, 2.5, 175.0), 2)

    # Active connected EVs estimate
    df["active_vehicles"] = np.maximum(1, np.round(df["charging_load_kw"] / np.random.uniform(11.0, 16.0, n)).astype(int))

    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    out_file = os.path.join("data", "ev_charging_data.csv")
    print(f"Generating EV charging dataset...")
    df = generate_ev_charging_dataset()
    df.to_csv(out_file, index=False)
    print(f"Successfully saved {len(df)} records to '{out_file}'.")
    print(df.head())
