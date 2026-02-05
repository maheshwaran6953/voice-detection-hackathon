import requests
import time

print("Waiting for deployment to complete...")
time.sleep(30)  # Wait 30 seconds for server to spin up

# Test with a simple base64 string (empty audio)
url = "https://voice-detection-api-8vsp.onrender.com/detect/"
headers = {
    "x-api-key": "hackathon-key-2024",
    "Content-Type": "application/json"
}
data = {
    "language": "en",
    "audio_format": "wav",
    "audio_base64_format": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQQAAAD//w=="
}

print("Testing deployed API...")
try:
    response = requests.post(url, headers=headers, json=data, timeout=10)
    print(f"✅ Deployment successful! Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ Test failed: {e}")
    print("Waiting a bit longer and trying again...")
    time.sleep(30)
    # Try again
    response = requests.post(url, headers=headers, json=data, timeout=10)
    print(f"Second attempt - Status: {response.status_code}")