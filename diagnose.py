import requests

urls = [
    "https://voice-detection-api-8vsp.onrender.com/",
    "https://voice-detection-api-8vsp.onrender.com/detect/test",
    "https://voice-detection-api-8vsp.onrender.com/health"
]

for url in urls:
    try:
        response = requests.get(url, timeout=5)
        print(f"✅ {url}: {response.status_code} - {response.text[:50]}")
    except Exception as e:
        print(f"❌ {url}: {e}")

# Test POST
test_url = "https://voice-detection-api-8vsp.onrender.com/detect/"
test_data = {
    "language": "en",
    "audio_format": "wav",
    "audio_base64_format": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQQAAAD//w=="
}
headers = {
    "x-api-key": "hackathon-key-2024",
    "Content-Type": "application/json"
}

print(f"\nTesting POST to {test_url}")
try:
    response = requests.post(test_url, json=test_data, headers=headers, timeout=10)
    print(f"POST Status: {response.status_code}")
    print(f"POST Response: {response.text}")
except Exception as e:
    print(f"POST Error: {e}")