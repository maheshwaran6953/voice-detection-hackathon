# 🎤 Voice Detection API - HCL GUVI Hackathon

**AI vs Human Voice Detection System**

Detects whether a voice sample is AI-generated or spoken by a real human. Supports 5 languages: English, Tamil, Hindi, Malayalam, and Telugu.

---

## ⚡ Quick Start (2 Minutes)

### Option 1: Automatic Setup (Recommended)

```bash
python quick_start.py
```

This automatically:
1. ✓ Checks Python installation
2. ✓ Installs all dependencies
3. ✓ Starts the API server
4. ✓ Runs comprehensive tests

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API server (Terminal 1)
python run.py

# 3. Run tests (Terminal 2)
python comprehensive_test.py
```

---

## 📖 Documentation

**For detailed setup and usage instructions, see:**
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete guide with examples
- **[http://localhost:8000/docs](http://localhost:8000/docs)** - Interactive API docs (when server running)

---

## 🔑 API Overview

### Main Endpoint
```
POST /detect/

Required Header: X-API-Key: hackathon-key-2024

Request:
{
    "audio_url": "https://example.com/audio.mp3",
    "language": "en",
    "test_description": "Optional description"
}

Response:
{
    "status": "success",
    "result": "HUMAN",
    "confidence": 0.8234,
    "language": "en",
    "processing_time_ms": 145,
    "features_extracted": 8
}
```

### Other Endpoints
- `GET /` - API information
- `GET /health` - Server health check
- `GET /detect/test` - Detection endpoint status

---

## 🎯 Features

✅ **Dual Classification:** AI-Generated or Human  
✅ **5 Language Support:** en, ta, hi, ml, te  
✅ **Confidence Scores:** 0.0 to 1.0  
✅ **Fast Processing:** < 500ms average  
✅ **Multiple Audio Formats:** MP3, WAV, FLAC, OGG  
✅ **Base64 Support:** Direct audio data or URL  
✅ **Authentication:** API key protection  
✅ **Error Handling:** Comprehensive validation  

---

## 🧪 Testing

### Run All Tests
```bash
python comprehensive_test.py
```

Tests include:
- Authentication & authorization
- Human voice detection
- AI voice detection
- Multiple language support
- Response format validation
- Performance testing
- Error handling

### Test with cURL
```bash
curl -X POST http://localhost:8000/detect/ \
  -H "X-API-Key: hackathon-key-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "test-mode",
    "language": "en"
  }'
```

---

## 📁 Project Structure

```
voice-detection-hackathon/
├── api/
│   ├── main.py              # FastAPI app
│   ├── models.py            # Data models
│   └── routers/
│       └── detect.py        # Detection endpoint
├── core/
│   └── audio_processor.py   # Audio processing
├── ml/
│   └── voice_detector.py    # ML model
├── comprehensive_test.py    # Test suite
├── quick_start.py          # Auto setup
├── SETUP_GUIDE.md          # Detailed guide
├── requirements.txt        # Dependencies
└── run.py                  # Start server
```

---

## 🔧 Configuration

API keys (in `api/routers/detect.py`):
- `hackathon-key-2024` - Main key
- `test-key` - Testing
- `demo-key` - Demo

Edit `.env` file to customize settings.

---

## 📊 API Keys & Languages

### Available API Keys
- `hackathon-key-2024`
- `test-key`
- `demo-key`

### Supported Languages
| Code | Language |
|------|----------|
| en | English |
| ta | Tamil |
| hi | Hindi |
| ml | Malayalam |
| te | Telugu |

---

## 🚀 Deployment

### Local Development
```bash
python run.py
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 api.main:app
```

### Docker
```bash
docker build -t voice-detection .
docker run -p 8000:8000 voice-detection
```

---

## 📚 Example Usage

### Python
```python
import requests

headers = {"X-API-Key": "hackathon-key-2024"}
payload = {
    "audio_url": "test-mode",
    "language": "en"
}

response = requests.post(
    "http://localhost:8000/detect/",
    json=payload,
    headers=headers
)

print(response.json())
```

### cURL
```bash
curl -X POST http://localhost:8000/detect/ \
  -H "X-API-Key: hackathon-key-2024" \
  -H "Content-Type: application/json" \
  -d '{"audio_url":"test-mode","language":"en"}'
```

---

## ❓ Troubleshooting

**Issue: "Module not found"**
```bash
pip install -r requirements.txt
```

**Issue: "Port already in use"**
Edit `run.py` and change port to 8001

**Issue: "Connection refused"**
Ensure server is running with `python run.py`

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for more troubleshooting.

---

## 📞 Support

For detailed help:
1. Read [SETUP_GUIDE.md](SETUP_GUIDE.md)
2. Check API docs at `http://localhost:8000/docs`
3. Review error messages in terminal
4. Run tests: `python comprehensive_test.py`

---

## 🏆 Performance

| Metric | Value |
|--------|-------|
| Response Time | < 500ms |
| Max File Size | 10MB |
| Max Duration | 30s |
| Supported Formats | MP3, WAV, FLAC, OGG |
| Languages | 5 |

---

## ✅ Hackathon Checklist

- [ ] Server runs successfully
- [ ] All tests pass
- [ ] API authentication works
- [ ] Human voice detected correctly
- [ ] AI voice detected correctly
- [ ] All 5 languages supported
- [ ] JSON response format correct
- [ ] Confidence scores valid (0.0-1.0)
- [ ] Response time < 5 seconds
- [ ] API publicly accessible

---

**Made for HCL GUVI Hackathon 2024** 🚀
