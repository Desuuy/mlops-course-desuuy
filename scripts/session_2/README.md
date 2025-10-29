# Session 2: MLflow with Epoch-by-Epoch Metric Logging

This session demonstrates how to use MLflow for experiment tracking with detailed epoch-by-epoch metric logging for machine learning models.

## Features

### 1. **Epoch-by-Epoch Metric Logging**
- Custom `MLflowSGDRegressor` class that logs metrics for each training epoch
- Tracks MSE, MAE, R², and RMSE for every epoch
- Logs metrics to MLflow with step-based tracking
- Early stopping based on convergence criteria

### 2. **Training Curves Visualization**
- Automatic generation of training curves plots
- 4-panel visualization showing MSE, MAE, R², and RMSE over epochs
- High-resolution plots saved as MLflow artifacts

### 3. **Enhanced MLflow Integration**
- Experiment tracking with detailed parameter logging
- Model registry integration
- Artifact storage (data, models, plots)
- Step-based metric logging for time series visualization

## Files

- `training.py` - Main training script with epoch logging
- `eval.py` - Model evaluation script using MLflow Model Registry
- `demo_epoch_logging.py` - Quick demo script with reduced epochs
- `README.md` - This documentation

## Usage

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Run Full Training with Epoch Logging**
```bash
python scripts/session_2/training.py
```

### 3. **Run Quick Demo (50 epochs)**
```bash
python scripts/session_2/demo_epoch_logging.py
```

### 4. **Evaluate Model**
```bash
python scripts/session_2/eval.py
```

### 5. **View MLflow UI**
```bash
mlflow ui
```
Open http://localhost:5000 to see:
- Epoch-by-epoch metrics in the metrics tab
- Training curves in the artifacts section
- Model registry with registered models

## Key Features Demonstrated

### **Epoch Metrics Logging**
- Metrics logged every 10 epochs (configurable)
- Step-based logging for time series visualization
- Early stopping when convergence is reached
- Final epoch metrics summary

### **Training Curves**
- Automatic plot generation showing:
  - Mean Squared Error over epochs
  - Mean Absolute Error over epochs
  - R² Score over epochs
  - Root Mean Squared Error over epochs
- High-resolution plots saved as artifacts

### **MLflow Integration**
- Experiment: "housing_price_prediction"
- Model Registry: "housing_price_predictor"
- Artifacts: data, model, plots, training curves
- Metrics: epoch-level and final performance metrics

## Customization

### **Change Epoch Logging Frequency**
In `training.py`, modify the logging frequency:
```python
# Log to MLflow every N epochs
if (epoch + 1) % N == 0 or epoch == self.max_iter - 1:
    # Log metrics
```

### **Add More Metrics**
Extend the `MLflowSGDRegressor` class to track additional metrics:
```python
# Add custom metrics to epoch_metric dictionary
epoch_metric = {
    'epoch': epoch + 1,
    'mse': mse,
    'mae': mae,
    'r2': r2,
    'rmse': rmse,
    'custom_metric': your_custom_metric,  # Add here
}
```

### **Modify Training Curves**
Update the `create_training_curves` function to include additional plots or change the layout.

## Example Output

When running the training script, you'll see:
```
2024-01-XX XX:XX:XX INFO housing_mlflow - Epoch 10/5000: MSE=1234567.8901, MAE=987.6543, R²=0.1234, RMSE=1111.1111
2024-01-XX XX:XX:XX INFO housing_mlflow - Epoch 20/5000: MSE=123456.7890, MAE=876.5432, R²=0.2345, RMSE=222.2222
...
2024-01-XX XX:XX:XX INFO housing_mlflow - Training completed after 150 epochs
2024-01-XX XX:XX:XX INFO housing_mlflow - Final metrics - MSE: 12345.6789, R²: 0.9876
```

## MLflow UI Features

1. **Metrics Tab**: View epoch-by-epoch metrics as time series
2. **Artifacts Tab**: Download training curves plots
3. **Model Registry**: Access registered models for deployment
4. **Run Comparison**: Compare different training runs

This implementation provides comprehensive experiment tracking with detailed epoch-level insights for better model development and debugging.
