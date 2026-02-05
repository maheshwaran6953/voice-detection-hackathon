import requests
import json

url = "https://voice-detection-api-8vsp.onrender.com/detect"
headers = {
    "x-api-key": "hackathon-key-2024",
    "Content-Type": "application/json"
}

# Test with simple base64
test_data = {
    "language": "en",
    "audio_format": "wav",
    "audio_base64_format": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQQAAAD//w=="
}

print("Testing API...")
print(f"URL: {url}")
print(f"Headers: {headers}")
print(f"Data: {json.dumps(test_data, indent=2)}")

try:
    response = requests.post(url, headers=headers, json=test_data, timeout=30)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("\n✅ API IS WORKING! Ready for hackathon!")
        result = response.json()
        print(f"Result: {result['result']} with {result['confidence']*100:.1f}% confidence")
        
        # Test with your actual audio
        print("\n\nTesting with your actual audio file...")
        import base64
        with open("trial.wav", "rb") as f:
            actual_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Send first 10000 chars to test
        test_data2 = {
            "language": "en",
            "audio_format": "wav",
            "audio_base64_format": actual_base64[:10000] + "..."
        }
        
        response2 = requests.post(url, headers=headers, json=test_data2, timeout=60)
        print(f"Actual audio test - Status: {response2.status_code}")
        print(f"Response: {response2.text[:200]}...")
        
except Exception as e:
    print(f"\n❌ Error: {e}")