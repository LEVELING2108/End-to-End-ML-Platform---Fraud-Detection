"""
Load testing script for the Fraud Detection API.
Simulates high-volume traffic to measure latency and throughput.
"""
import time
import requests
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor

# Configuration
URL = "http://localhost:8000/predict"
EXPLAIN_URL = "http://localhost:8000/explain"
API_KEY = "fraud-detection-secret-key"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-KEY": API_KEY
}
PAYLOAD = {
    "amount": 150.0,
    "hour": 14,
    "day_of_week": 3,
    "merchant_category": "grocery"
}

def make_request(url):
    start = time.perf_counter()
    try:
        response = requests.post(url, json=PAYLOAD, headers=HEADERS, timeout=5)
        latency = time.perf_counter() - start
        return latency, response.status_code
    except Exception as e:
        return time.perf_counter() - start, 500

def run_benchmark(url, name, num_requests=100, max_workers=10):
    print(f"\n--- Benchmarking {name} ({num_requests} requests, {max_workers} concurrent workers) ---")
    
    latencies = []
    status_codes = []
    
    start_time = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda _: make_request(url), range(num_requests)))
    
    total_time = time.perf_counter() - start_time
    
    for lat, status in results:
        latencies.append(lat)
        status_codes.append(status)
    
    success_rate = status_codes.count(200) / len(status_codes)
    avg_latency = statistics.mean(latencies) * 1000 # to ms
    p95_latency = statistics.quantiles(latencies, n=20)[18] * 1000 # 95th percentile
    rps = num_requests / total_time
    
    print(f"Results for {name}:")
    print(f"  Throughput:    {rps:.2f} requests/sec")
    print(f"  Avg Latency:   {avg_latency:.2f} ms")
    print(f"  P95 Latency:   {p95_latency:.2f} ms")
    print(f"  Success Rate:  {success_rate:.1%}")

if __name__ == "__main__":
    # Test basic prediction
    run_benchmark(URL, "Basic Prediction (/predict)", num_requests=200, max_workers=20)
    
    # Test explainability (more compute intensive)
    run_benchmark(EXPLAIN_URL, "Explainable Prediction (/explain)", num_requests=50, max_workers=5)
