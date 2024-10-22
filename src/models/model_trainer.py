from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pandas as pd
import numpy as np

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=random_state
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state
            )
        }
        self.trained_models = {}
        self.predictions = {}
        
    def prepare_data(self, X, y, test_size=0.2):
        """Split and scale the data"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X, y):
        """Train all models"""
        X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data(X, y)
        results = {}
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            self.trained_models[name] = model
            
            # Make predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            self.predictions[name] = test_pred
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'train_predictions': train_pred,
                'test_predictions': test_pred,
                'y_test': y_test,
                'cv_scores': cv_scores
            }
        
        return results
    
    def get_feature_importance(self, model_name, feature_names):
        """Get feature importance for a specific model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
            
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = abs(model.coef_)
        else:
            raise ValueError(f"Model {model_name} doesn't support feature importance")
            
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)