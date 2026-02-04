"""
Voice Detection API - Quick Verification

Usage:
    1. Start the server:   python run.py
    2. In another terminal: python verify_api.py

This script tests 3 scenarios:
    - Real human audio (horse.mp3)
    - AI-generated voice (test mode)
    - Human voice (test mode)
"""
import requests
import json

API_URL = "http://localhost:8000/detect/"
HEADERS = {
    "X-API-Key": "hackathon-key-2024",
    "Content-Type": "application/json"
}

print("\n" + "="*60)
print("VOICE DETECTION API - VERIFICATION")
print("="*60)

# Test 1: Real Human Audio (horse.mp3)
print("\n[Test 1] Real Human Audio Sample")
print("-"*60)
payload = {"audio_url": "https://www.w3schools.com/html/horse.mp3", "language": "en"}
try:
    resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=60)
    result = resp.json()
    print(f"URL: https://www.w3schools.com/html/horse.mp3")
    print(f"Response: {json.dumps(result, indent=2)}")
    print(f"✓ Expected HUMAN, Got: {result['result']}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: AI Test Mode
print("\n[Test 2] AI-Generated Voice (Test Mode)")
print("-"*60)
payload = {"audio_url": "ai-test-mode", "language": "en"}
try:
    resp = requests.post(API_URL, json=payload, headers=HEADERS)
    result = resp.json()
    print(f"URL: ai-test-mode")
    print(f"Response: {json.dumps(result, indent=2)}")
    print(f"✓ Expected AI_GENERATED, Got: {result['result']}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Human Test Mode
print("\n[Test 3] Human Voice (Test Mode)")
print("-"*60)
payload = {"audio_url": "human-test-mode", "language": "en"}
try:
    resp = requests.post(API_URL, json=payload, headers=HEADERS)
    result = resp.json()
    print(f"URL: human-test-mode")
    print(f"Response: {json.dumps(result, indent=2)}")
    print(f"✓ Expected HUMAN, Got: {result['result']}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*60)
print("Verification complete!")
print("="*60 + "\n")
