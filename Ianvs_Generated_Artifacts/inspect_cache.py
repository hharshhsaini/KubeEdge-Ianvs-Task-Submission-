import json
import sys

cache_file = sys.argv[1]
with open(cache_file, 'r') as f:
    data = json.load(f)

for i, entry in enumerate(data):
    print(f"\n=== Cache Entry {i} ===")
    print(f"Config: {json.dumps(entry['config'], indent=2)}")
    print(f"Number of cached results: {len(entry.get('result', []))}")
