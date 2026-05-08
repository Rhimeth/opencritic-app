from data import OpenCriticDataClient

client = OpenCriticDataClient()
print("Refreshing data (2 pages)...")
stats = client.refresh(pages=2, page_size=10)
print(f"Stats: {stats}")

df = client.to_dataframe()
print(f"\nDataFrame shape: {df.shape}")
print(df.head())