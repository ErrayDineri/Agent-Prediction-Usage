import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import json
import os

# === Load and prepare the data ===
with open("metric_monthly_generated.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Convert data types
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['cpu_usage'] = df['cpu_usage'].astype(float)
df['memory_usage'] = df['memory_usage'].astype(float)
df['latency'] = df['latency'].astype(float)

# Filter for the last 30 days
latest_time = df['timestamp'].max()
one_month_ago = latest_time - timedelta(days=30)
df = df[df['timestamp'] >= one_month_ago]

# Sort by VM and time
df = df.sort_values(by=["VM Name", "timestamp"])
vm_names = df['VM Name'].unique()

# === Feature Engineering ===
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['weekday'] = df['timestamp'].dt.weekday

def add_lag_features(df, lag_hours=[1, 2, 3]):
    for lag in lag_hours:
        df[f'cpu_lag_{lag}'] = df.groupby('VM Name')['cpu_usage'].shift(lag)
        df[f'mem_lag_{lag}'] = df.groupby('VM Name')['memory_usage'].shift(lag)
        df[f'lat_lag_{lag}'] = df.groupby('VM Name')['latency'].shift(lag)
    return df

df = add_lag_features(df)
df = df.dropna()

# === Model Training ===
features = [
    'hour', 'day', 'weekday',
    'cpu_lag_1', 'cpu_lag_2', 'cpu_lag_3',
    'mem_lag_1', 'mem_lag_2', 'mem_lag_3',
    'lat_lag_1', 'lat_lag_2', 'lat_lag_3'
]

# Prepare training data
X = df[features]
y_cpu = df['cpu_usage']
y_mem = df['memory_usage']
y_lat = df['latency']

# Time series split
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_cpu_train, y_cpu_test = y_cpu[:split_idx], y_cpu[split_idx:]
y_mem_train, y_mem_test = y_mem[:split_idx], y_mem[split_idx:]
y_lat_train, y_lat_test = y_lat[:split_idx], y_lat[split_idx:]

# Train models
cpu_model = RandomForestRegressor(n_estimators=100, random_state=42)
cpu_model.fit(X_train, y_cpu_train)

mem_model = RandomForestRegressor(n_estimators=100, random_state=42)
mem_model.fit(X_train, y_mem_train)

lat_model = RandomForestRegressor(n_estimators=100, random_state=42)
lat_model.fit(X_train, y_lat_train)

# === Evaluation ===
print(f"[INFO] CPU MSE: {mean_squared_error(y_cpu_test, cpu_model.predict(X_test)):.2f}")
print(f"[INFO] MEM MSE: {mean_squared_error(y_mem_test, mem_model.predict(X_test)):.2f}")
print(f"[INFO] LAT MSE: {mean_squared_error(y_lat_test, lat_model.predict(X_test)):.2f}")

# === Forecasting next 24 hours ===
forecast_horizon = 24
results = []

for vm in vm_names:
    history = df[df["VM Name"] == vm].copy()
    last_row = history.iloc[-1].copy()

    for step in range(1, forecast_horizon + 1):
        future_time = last_row['timestamp'] + pd.Timedelta(hours=1)

        input_row = {
            'hour': future_time.hour,
            'day': future_time.day,
            'weekday': future_time.weekday(),
            'cpu_lag_1': last_row['cpu_usage'],
            'cpu_lag_2': last_row['cpu_lag_1'],
            'cpu_lag_3': last_row['cpu_lag_2'],
            'mem_lag_1': last_row['memory_usage'],
            'mem_lag_2': last_row['mem_lag_1'],
            'mem_lag_3': last_row['mem_lag_2'],
            'lat_lag_1': last_row['latency'],
            'lat_lag_2': last_row['lat_lag_1'],
            'lat_lag_3': last_row['lat_lag_2'],
        }

        input_df = pd.DataFrame([input_row])
        pred_cpu = cpu_model.predict(input_df)[0]
        pred_mem = mem_model.predict(input_df)[0]
        pred_lat = lat_model.predict(input_df)[0]

        results.append({
            "VM Name": vm,
            "timestamp": future_time,
            "predicted_cpu": pred_cpu,
            "predicted_memory": pred_mem,
            "predicted_latency": pred_lat
        })

        # Update for next step
        last_row['timestamp'] = future_time
        last_row['cpu_lag_3'] = last_row['cpu_lag_2']
        last_row['cpu_lag_2'] = last_row['cpu_lag_1']
        last_row['cpu_lag_1'] = pred_cpu
        last_row['cpu_usage'] = pred_cpu

        last_row['mem_lag_3'] = last_row['mem_lag_2']
        last_row['mem_lag_2'] = last_row['mem_lag_1']
        last_row['mem_lag_1'] = pred_mem
        last_row['memory_usage'] = pred_mem

        last_row['lat_lag_3'] = last_row['lat_lag_2']
        last_row['lat_lag_2'] = last_row['lat_lag_1']
        last_row['lat_lag_1'] = pred_lat
        last_row['latency'] = pred_lat

# === Display results ===
for row in results:
    print(f"{row['VM Name']} @ {row['timestamp']:%Y-%m-%d %H:%M} → "
        f"CPU: {row['predicted_cpu']:.2f}%, "
        f"MEM: {row['predicted_memory']:.2f}%, "
        f"LAT: {row['predicted_latency']:.2f}ms")
import joblib
if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(cpu_model, 'models/cpu_model.joblib')
joblib.dump(mem_model, 'models/mem_model.joblib')
joblib.dump(lat_model, 'models/lat_model.joblib')
