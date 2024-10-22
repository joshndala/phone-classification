from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

def create_metrics_df(results):
    """Create DataFrame with all model metrics"""
    metrics_data = []
    
    for model_name, result in results.items():
        metrics = calculate_metrics(
            result['y_test'],
            result['test_predictions']
        )
        
        # Add each metric to the data
        for metric_name, value in metrics.items():
            metrics_data.append({
                'Model': model_name,
                'Metric': metric_name,
                'Value': value
            })
    
    return pd.DataFrame(metrics_data)

def print_model_summary(results):
    """Print summary of model performance"""
    for model_name, result in results.items():
        metrics = calculate_metrics(
            result['y_test'],
            result['test_predictions']
        )
        cv_scores = result['cv_scores']
        
        print(f"\n{model_name} Results:")
        print(f"RMSE: ${metrics['RMSE']:.2f}")
        print(f"MAE: ${metrics['MAE']:.2f}")
        print(f"R² Score: {metrics['R²']:.3f}")
        print(f"CV R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")