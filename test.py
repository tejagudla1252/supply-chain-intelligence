import pandas as pd
df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='latin-1')
print(df.shape)          # (180519, 53)
print(df.columns.tolist())
print(df['Late_delivery_risk'].value_counts())
print(df['Order Region'].unique())