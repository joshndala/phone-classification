import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_storage(data: pd.DataFrame) -> pd.DataFrame:
    """Convert 1TB to 1024GB in storage column"""
    data = data.copy()
    data.loc[data['storage(GB)'] == 1.0, 'storage(GB)'] = 1024.0
    return data

def split_resolution(data: pd.DataFrame) -> pd.DataFrame:
    """Split resolution into width and height"""
    data = data.copy()
    data[['width', 'height']] = data['resolution'].str.split('x', expand=True).astype('int64')
    return data

def encode_brands(data: pd.DataFrame) -> pd.DataFrame:
    """Encode brand names"""
    data = data.copy()
    label_encoder = LabelEncoder()
    data['brand_encoded'] = label_encoder.fit_transform(data['brand'])
    return data

def extract_year(data: pd.DataFrame) -> pd.DataFrame:
    """Extract year from announcement date"""
    data = data.copy()
    data['announcement_year'] = data['announcement_date'].apply(lambda x: x.split('-')[0]).astype('int64')
    return data

def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns"""
    columns_to_drop = ['phone_name', 'brand', 'os', 'resolution', 'battery_type', 'announcement_date']
    return data.drop(columns=columns_to_drop)

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Execute all cleaning operations"""
    data = clean_storage(data)
    data = split_resolution(data)
    data = encode_brands(data)
    data = extract_year(data)
    data = drop_columns(data)
    return data
