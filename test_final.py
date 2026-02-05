import requests
import base64
import json

# Read your audio file
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

print("Testing API with your audio file...")
print(f"Base64 length: {len(base64_string)} characters")

try:
    response = requests.post(url, headers=headers, json=data, timeout=60)
    print(f"\n✅ Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("🎉 SUCCESS! API is working!")
        print("\nResponse:")
        print(json.dumps(response.json(), indent=2))
        
        # Check if response has the right structure
        json_response = response.json()
        if "result" in json_response and "confidence" in json_response:
            print(f"\n✅ Perfect! Your API returns: {json_response['result']} with {json_response['confidence']*100:.1f}% confidence")
        else:
            print("⚠️ Response structure might be different than expected")
    else:
        print(f"❌ Error: {response.text}")
        
except requests.exceptions.Timeout:
    print("❌ Request timed out (60 seconds)")
except Exception as e:
    print(f"❌ Error: {e}")