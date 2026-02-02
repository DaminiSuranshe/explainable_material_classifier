"""
Simple concurrent load test.
"""

import requests
import threading

URL = "http://localhost:8000/health"


def hit():
    for _ in range(10):
        requests.get(URL)


threads = [threading.Thread(target=hit) for _ in range(20)]

for t in threads:
    t.start()

for t in threads:
    t.join()

print("Load test complete")
