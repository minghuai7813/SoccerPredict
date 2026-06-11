"""Quick test of FotMob API."""
from curl_cffi import requests as cr
import json

resp = cr.get("https://www.fotmob.com/api/leagues?id=71", impersonate="chrome", timeout=15)
print(f"Status: {resp.status_code}")
ct = resp.headers.get("content-type", "?")
print(f"Content-Type: {ct}")
if resp.status_code == 200:
    d = resp.json()
    print(f"Top keys: {list(d.keys())[:10]}")
    if "stats" in d:
        stats = d["stats"]
        print(f"Stats type: {type(stats)}")
        if isinstance(stats, dict):
            print(f"Stats keys: {list(stats.keys())[:10]}")
        elif isinstance(stats, list):
            print(f"Stats list len: {len(stats)}")
            if stats:
                print(f"First stat: {json.dumps(stats[0], indent=2)[:500]}")
    if "table" in d:
        print("Has table data")
else:
    print(resp.text[:500])
