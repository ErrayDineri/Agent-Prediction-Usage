# Agent Prediction Usage Project

This project provides a pipeline for time series prediction of VM metrics (CPU, memory, latency) and integrates with n8n for workflow automation and Telegram notifications.

## Setup Instructions

1. **Clone the repository**
   ```powershell
   git clone <github_link>
   cd Agent Prediction Usage
   ```

2. **Install Python dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Install n8n**
   - Follow the official n8n installation guide: https://docs.n8n.io/hosting/installation/

4. **Import the n8n workflow**
   - Open n8n in your browser.
   - Import the workflow from the file `n8n_workflow.json`.

5. **Configure the Telegram node**
   - In the imported workflow, set your Telegram bot credentials and c hat ID in the Telegram node.

6. **Train the prediction models**
   ```powershell
   python train.py
   ```
   - This will generate the model files in the `models/` directory.

7. **Start the FastAPI server**
   ```powershell
   uvicorn main:app --reload
   ```
   - The API will be available at `http://127.0.0.1:8000` by default.

8. **Test the n8n workflow**
   - Trigger the workflow manually in n8n to test the integration.

   **Important:** For testing, the workflow is set to trigger on click. For production, change the trigger to a cron task for scheduled execution.

## Notes & Recommendations for Improvement

- **Data Generation:**
  - The current synthetic data generation uses random uniform values. For more realistic results, consider using probabilistic laws (e.g., normal, Poisson, or empirical distributions) that better reflect real-world VM usage patterns.
  - If possible, use actual historical data from your company to train and validate the models for more accurate and relevant predictions.

- **Model Choice:**
  - The current training and inference scripts use RandomForestRegressor, which does not inherently capture temporal dependencies in time series data.
  - For improved forecasting, consider using models designed for time series, such as ARIMA, Prophet, or deep learning models like LSTM or Temporal Fusion Transformers.

- **Workflow Trigger:**
  - The n8n workflow is set to trigger manually for testing. For production, change the trigger to a cron task for scheduled, automated execution.

---

Feel free to open issues or contribute improvements!
