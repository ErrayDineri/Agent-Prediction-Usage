from fastapi import FastAPI, HTTPException
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os

app = FastAPI()

# Globals
cpu_model = None
mem_model = None
lat_model = None
DATA_PATH = "metric_monthly2.csv"


def add_lag_features(df, lag_hours=[1, 2, 3]):
    for lag in lag_hours:
        df[f'cpu_lag_{lag}'] = df.groupby('VM Name')['cpu_usage'].shift(lag)
        df[f'mem_lag_{lag}'] = df.groupby('VM Name')['memory_usage'].shift(lag)
        df[f'lat_lag_{lag}'] = df.groupby('VM Name')['latency'].shift(lag)
    return df


@app.on_event("startup")
def load_models():
    global cpu_model, mem_model, lat_model
    cpu_model = joblib.load('models/cpu_model.joblib')
    mem_model = joblib.load('models/mem_model.joblib')
    lat_model = joblib.load('models/lat_model.joblib')
    print("[INFO] Models loaded successfully.")


@app.get("/predict")
def predict_for_today():
    global cpu_model, mem_model, lat_model

    if not (cpu_model and mem_model and lat_model):
        raise HTTPException(status_code=500, detail="Models not loaded")

    if not os.path.isfile(DATA_PATH):
        raise HTTPException(status_code=500, detail=f"Data file '{DATA_PATH}' not found")

    df = pd.read_csv(DATA_PATH)

    # Preprocess
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['cpu_usage'] = df['cpu_usage'].astype(float)
    df['memory_usage'] = df['memory_usage'].astype(float)
    df['latency'] = df['latency'].astype(float)

    # Keep last 30 days
    latest_time = df['timestamp'].max()
    one_month_ago = latest_time - timedelta(days=30)
    df = df[df['timestamp'] >= one_month_ago]

    df = df.groupby(['VM Name', 'timestamp'], as_index=False).agg({
        'cpu_usage': 'mean',
        'memory_usage': 'mean',
        'latency': 'mean'
    })

    df = df.sort_values(by=['VM Name', 'timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday

    df = add_lag_features(df)
    df = df.dropna()

    features = [
        'hour', 'day', 'weekday',
        'cpu_lag_1', 'cpu_lag_2', 'cpu_lag_3',
        'mem_lag_1', 'mem_lag_2', 'mem_lag_3',
        'lat_lag_1', 'lat_lag_2', 'lat_lag_3'
    ]

    today = datetime.now().replace(minute=0, second=0, microsecond=0)
    target_hours = [today.replace(hour=h) for h in range(24)]

    results = []
    vm_names = df['VM Name'].unique()

    for vm in vm_names:
        history = df[df['VM Name'] == vm].copy()
        last_row = history.iloc[-1].copy()

        for target_time in target_hours:
            input_row = {
                'hour': target_time.hour,
                'day': target_time.day,
                'weekday': target_time.weekday(),
                'cpu_lag_1': last_row['cpu_usage'],
                'cpu_lag_2': last_row['cpu_lag_1'],
                'cpu_lag_3': last_row['cpu_lag_2'],
                'mem_lag_1': last_row['mem_lag_1'],
                'mem_lag_2': last_row['mem_lag_2'],
                'mem_lag_3': last_row['mem_lag_3'],
                'lat_lag_1': last_row['lat_lag_1'],
                'lat_lag_2': last_row['lat_lag_2'],
                'lat_lag_3': last_row['lat_lag_3'],
            }

            input_df = pd.DataFrame([input_row], columns=features)

            pred_cpu = cpu_model.predict(input_df)[0]
            pred_mem = mem_model.predict(input_df)[0]
            pred_lat = lat_model.predict(input_df)[0]

            results.append({
                "VM Name": vm,
                "timestamp": target_time.isoformat(),
                "predicted_cpu": round(pred_cpu, 2),
                "predicted_memory": round(pred_mem, 2),
                "predicted_latency": round(pred_lat, 2)
            })

            # Update last_row for next hour
            last_row['timestamp'] = target_time
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

    from collections import defaultdict

    grouped_results = defaultdict(list)
    for entry in results:
        if entry["predicted_cpu"] >80 or entry["predicted_memory"] >75 or entry["predicted_latency"] >120:
            vm = entry["VM Name"]
            grouped_results[vm].append({
                "timestamp": entry["timestamp"],
                "predicted_cpu": entry["predicted_cpu"],
                "predicted_memory": entry["predicted_memory"],
                "predicted_latency": entry["predicted_latency"]
            })

    return {"predictions": dict(grouped_results)}
