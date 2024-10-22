import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_screen_features(data):
    """Create screen-related features"""
    data = data.copy()
    data['screen_area'] = data['width'] * data['height']
    data['pixel_density'] = data['screen_area'] / (data['inches'] ** 2)
    return data

def create_video_scores(data):
    """Calculate video capability scores"""
    data = data.copy()
    video_res_weights = {
        'video_720p': 1, 'video_1080p': 2,
        'video_4K': 4, 'video_8K': 8
    }
    video_fps_weights = {
        'video_30fps': 1, 'video_60fps': 2,
        'video_120fps': 3, 'video_240fps': 4,
        'video_480fps': 5, 'video_960fps': 6
    }
    
    data['video_res_score'] = sum(data[col] * video_res_weights[col]
                                 for col in video_res_weights.keys())
    data['video_fps_score'] = sum(data[col] * video_fps_weights[col]
                                 for col in video_fps_weights.keys())
    data['video_total_score'] = data['video_res_score'] * data['video_fps_score']
    return data

def create_memory_features(data):
    """Create memory-related features"""
    data = data.copy()
    data['memory_ratio'] = data['ram(GB)'] / data['storage(GB)']
    data['memory_total'] = data['ram(GB)'] + data['storage(GB)']
    data['ram_storage_interaction'] = data['ram(GB)'] * np.log1p(data['storage(GB)'])
    return data

def create_battery_features(data):
    """Create battery-related features"""
    data = data.copy()
    data['battery_per_inch'] = data['battery'] / data['inches']
    data['battery_per_weight'] = data['battery'] / data['weight(g)']
    return data

def create_generation_features(data, current_year=2024):
    """Create generation and age-related features"""
    data = data.copy()
    data['device_age'] = current_year - data['announcement_year']
    data['is_recent_gen'] = data['device_age'] <= 2
    data['generation_score'] = np.exp(-data['device_age'] * 0.5)
    return data

def create_composite_scores(data):
    """Create composite score features"""
    data = data.copy()
    
    # Normalize numerical features first
    scaler = MinMaxScaler()
    numerical_cols = ['battery', 'ram(GB)', 'storage(GB)', 'weight(g)', 'inches']
    for col in numerical_cols:
        data[f'{col}_normalized'] = scaler.fit_transform(data[[col]])
    
    # Create composite scores
    data['specs_score'] = (
        data['ram(GB)'] * 0.3 +
        np.log1p(data['storage(GB)']) * 0.2 +
        data['video_total_score'] * 0.2 +
        data['battery'] / 1000 * 0.15 +
        data['screen_area'] / 100000 * 0.15
    )
    
    data['performance_score'] = (
        data['ram(GB)_normalized'] * 0.4 +
        data['storage(GB)_normalized'] * 0.3 +
        data['generation_score'] * 0.3
    )
    
    return data

def create_price_features(data):
    """Create price-related features"""
    data = data.copy()
    data['price_segment'] = pd.qcut(data['price(USD)'],
                                  q=5,
                                  labels=['budget', 'low_mid', 'mid', 'high_mid', 'premium'])
    
    brand_avg_price = data.groupby('brand_encoded')['price(USD)'].transform('mean')
    data['price_vs_brand_avg'] = data['price(USD)'] / brand_avg_price
    return data

def engineer_features(data):
    """Execute all feature engineering operations"""
    data = create_screen_features(data)
    data = create_video_scores(data)
    data = create_memory_features(data)
    data = create_battery_features(data)
    data = create_generation_features(data)
    data = create_composite_scores(data)
    data = create_price_features(data)
    return data