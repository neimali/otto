import pandas as pd

m=pd.read_parquet('/home/qiaodawang19/otto/data/memoryopt/train.parquet')
print(m.head(10))