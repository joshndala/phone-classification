import pandas as pd

def load_data(url: str) -> pd.DataFrame:
    """
    Load smartphone data from URL
    
    Args:
        url: URL to raw data
        
    Returns:
        pandas DataFrame with loaded data
    """
    data = pd.read_csv(url)
    return data

def get_basic_info(data: pd.DataFrame) -> None:
    """Print basic information about the dataset"""
    print(data.info())
    print("\nSample of data:")
    print(data.head())
    print("\nBasic statistics:")
    print(data.describe())

def get_value_counts(data: pd.DataFrame, columns: list) -> dict:
    """Get value counts for specified columns"""
    return {col: data[col].value_counts() for col in columns}
