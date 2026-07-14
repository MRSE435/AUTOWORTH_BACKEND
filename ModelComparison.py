# Run this ONCE, separately, not as part of your live Flask app
import json

model_comparison_data = {
    "models": [
        {"name": "Linear Regression", "test_r2": 0.9053, "train_r2": 0.91545, "mae": 116218.35},
        {"name": "Ridge", "test_r2":  0.9062, "train_r2": 0.9301, "mae": 118288},
        {"name": "Lasso", "test_r2":0.8968, "train_r2": 0.8625 , "mae":123782},
        {"name": "Random Forest", "test_r2": 0.9356, "train_r2": 0.9535, "mae":  100200.63},
        {"name": "XGBoost", "test_r2": 0.9427, "train_r2": 0.9589, "mae": 91985.12},
    ]
}

with open('model_comparison.json', 'w') as f:
    json.dump(model_comparison_data, f, indent=2)