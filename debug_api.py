from opencritic_app.data import OpenCriticDataClient
import requests
import json

client = OpenCriticDataClient()
params = {"skip": 0, "limit": 5, "sort": "date"}
response = requests.get(client.base_url, headers=client.headers, params=params)
data = response.json()
for game in data:
    print(f"Title: {game.get('name')}")
    print(f"Companies: {game.get('Companies')}")
    print("-" * 40)