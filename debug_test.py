import requests
import json

url = "https://voice-detection-api-8vsp.onrender.com/detect"
headers = {
    "x-api-key": "hackathon-key-2024",
    "Content-Type": "application/json"
}

# Test what the API actually receives
test_data = {
    "language": "en",
    "audio_format": "wav",
    "audio_base64_format": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQQAAAD//w=="
}

print("Sending:", json.dumps(test_data, indent=2))

response = requests.post(url, headers=headers, json=test_data)
print(f"\nStatus: {response.status_code}")
print(f"Response: {response.text}")

# Also try with the original 'audio' field
print("\n\nTrying with 'audio' field instead:")
test_data2 = {
    "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQQAAAD//w==",
    "language": "en"
}

response2 = requests.post(url, headers=headers, json=test_data2)
print(f"Status: {response2.status_code}")
print(f"Response: {response2.text}")