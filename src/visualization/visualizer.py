import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, save_path=None):
        """
        Initialize visualizer
        
        Args:
            save_path: Optional path to save visualizations
        """
        self.save_path = save_path
        self.set_style()
    
    @staticmethod
    def set_style():
        """Set default style for plots"""
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def save_or_show(self, plt, name=None):
        """Either save the plot or display it"""
        if self.save_path and name:
            plt.savefig(f"{self.save_path}/{name}.png")
            plt.close()
        else:
            plt.show(block=False)
            plt.pause(1)
            plt.close()
    
    def plot_correlation_matrix(self, data, title="Correlation Matrix"):
        """Plot correlation matrix heatmap"""
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=['int64', 'float64'])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(title)
        self.save_or_show(plt, "correlation_matrix")
    
    def plot_price_distribution(self, data):
        """Plot price distribution"""
        plt.figure(figsize=(14, 6))
        sns.histplot(data['price(USD)'], kde=True)
        plt.title('Price Distribution')
        plt.xlabel('Price (USD)')
        self.save_or_show(plt, "price_distribution")
    
    def plot_feature_importance(self, importance_df, title="Feature Importance"):
        """Plot feature importance"""
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=importance_df.head(10))
        plt.title(title)
        self.save_or_show(plt, "feature_importance")
    
    def plot_predictions_vs_actual(self, results, models):
        """Plot predictions vs actual for all models"""
        plt.figure(figsize=(15, 5))
        
        for i, model_name in enumerate(models, 1):
            plt.subplot(1, 3, i)
            y_test = results[model_name]['y_test']
            predictions = results[model_name]['test_predictions']
            
            plt.scatter(y_test, predictions, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2)
            plt.xlabel('Actual Price')
            plt.ylabel('Predicted Price')
            plt.title(f'{model_name}')
        
        plt.tight_layout()
        self.save_or_show(plt, "predictions_vs_actual")
    
    def plot_model_metrics(self, metrics_df):
        """Plot model performance metrics"""
        plt.figure(figsize=(12, 6))
        
        g = sns.barplot(
            data=metrics_df,
            x='Metric',
            y='Value',
            hue='Model',
            palette=['#2ecc71', '#3498db', '#e74c3c']
        )
        
        plt.title('Model Performance Comparison', fontsize=14, pad=15)
        plt.xlabel('Metric', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        
        for container in g.containers:
            g.bar_label(container, fmt='%.3f', padding=3)
        
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        self.save_or_show(plt, "model_metrics")

    def plot_all(self, data, results, metrics_df, importance_df, models):
        """Plot all visualizations at once"""
        try:
            self.plot_correlation_matrix(data)
            self.plot_price_distribution(data)
            self.plot_feature_importance(importance_df)
            self.plot_predictions_vs_actual(results, models)
            self.plot_model_metrics(metrics_df)
        except Exception as e:
            print(f"Error during visualization: {str(e)}")