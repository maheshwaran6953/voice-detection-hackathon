import requests
import base64

# Read your trial.wav file
with open("trial.wav", "rb") as f:
    base64_string = base64.b64encode(f.read()).decode('utf-8')

url = "https://voice-detection-api-8vsp.onrender.com/detect/"
headers = {
    "x-api-key": "hackathon-key-2024",
    "Content-Type": "application/json"
}
data = {
    "language": "en",
    "audio_format": "wav",
    "audio_base64_format": base64_string
}

print("Sending request to API...")
print(f"Base64 string length: {len(base64_string)} characters")

try:
    response = requests.post(url, headers=headers, json=data, timeout=30)
    print(f"\n✅ Status: {response.status_code}")
    print(f"Response: {response.text}")
except requests.exceptions.Timeout:
    print("❌ Request timed out!")
except Exception as e:
    print(f"❌ Error: {e}")