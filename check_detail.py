from opencritic_app.data import OpenCriticDataClient
import requests
import json

# Get a sample game ID from your database
client = OpenCriticDataClient()
client._load_database()
if client.records:
    sample_id = next(iter(client.records.keys()))
    print(f"Checking game ID: {sample_id}")
    url = f"{client.base_url}/{sample_id}"
    response = requests.get(url, headers=client.headers)
    if response.status_code == 200:
        data = response.json()
        print("Keys in detail response:", list(data.keys()))
        print("Companies:", data.get('Companies'))
        # Print first 2000 chars of response to see structure
        print(json.dumps(data, indent=2)[:2000])
    else:
        print(f"Failed: {response.status_code}")
else:
    print("No games in database. Run refresh first.")