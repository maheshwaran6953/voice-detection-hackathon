import requests
import base64
import json

def test_api():
    url = "https://voice-detection-api-8vsp.onrender.com/detect"
    headers = {
        "x-api-key": "hackathon-key-2024",
        "Content-Type": "application/json"
    }
    
    # Test 1: Simple base64
    print("Test 1: Simple base64")
    data1 = {
        "language": "en",
        "audio_format": "wav",
        "audio_base64_format": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQQAAAD//w=="
    }
    
    try:
        r1 = requests.post(url, headers=headers, json=data1, timeout=30)
        print(f"Status: {r1.status_code}")
        print(f"Response: {r1.text}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Test 2: With your actual audio (first part only)
    print("Test 2: With your actual audio")
    try:
        with open("trial.wav", "rb") as f:
            full_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        data2 = {
            "language": "en",
            "audio_format": "wav",
            "audio_base64_format": full_base64[:50000]  # First 50K chars for testing
        }
        
        r2 = requests.post(url, headers=headers, json=data2, timeout=60)
        print(f"Status: {r2.status_code}")
        print(f"Response: {r2.text[:200]}...\n")
    except Exception as e:
        print(f"Error: {e}\n")

if __name__ == "__main__":
    print("Waiting for deployment... (sleeping 120 seconds)")
    import time
    time.sleep(120)  # Wait 2 minutes for deployment
    test_api()