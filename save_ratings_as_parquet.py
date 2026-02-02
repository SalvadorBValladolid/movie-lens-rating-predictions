import pandas as pd

dtypes = {
    'userId': 'int32',
    'movieId': 'int32',
    'rating': 'float32'
}

df = pd.read_csv(
    'data/rating.csv',
    usecols=['userId', 'movieId', 'rating', 'timestamp'],
    dtype=dtypes,
    parse_dates=['timestamp']
)

df.to_parquet(
    "data/rating.parquet",
    engine="pyarrow",
    compression="snappy"
)