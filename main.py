from src.data.data_loader import load_data, get_basic_info
from src.preprocessing.data_cleaner import clean_data
from src.features.feature_engineer import engineer_features
from src.models.model_trainer import ModelTrainer
from src.visualization.visualizer import Visualizer
from src.utils.metrics import print_model_summary, create_metrics_df
import os

def main():
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Initialize visualizer with save path
    viz = Visualizer(save_path='visualizations')
    
    # Load and process data
    url = 'https://raw.githubusercontent.com/joshndala/phone-classification/refs/heads/main/cleaned_all_phones.csv'
    data = load_data(url)
    get_basic_info(data)
    
    # Clean and engineer features
    cleaned_data = clean_data(data)
    final_data = engineer_features(cleaned_data)
    
    # Prepare features and target
    X = final_data.drop(['price(USD)', 'price_segment'], axis=1)
    y = final_data['price(USD)']
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_models(X, y)
    
    # Print results
    print_model_summary(results)
    
    # Create metrics DataFrame
    metrics_df = create_metrics_df(results)
    
    # Get feature importance
    importance_df = trainer.get_feature_importance('Random Forest', X.columns)

    # Initialize visualizer with save path
    viz = Visualizer(save_path='visualization')
    
    # Create all visualizations
    viz.plot_all(final_data, results, metrics_df, importance_df, trainer.models.keys())

if __name__ == "__main__":
    main()