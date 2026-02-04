# HCL Guvi Hackathon - Voice Detection API

## API Endpoint
`POST /detect/`

## Required Headers
- `X-API-Key: hackathon-key-2024`
- `Content-Type: application/json`

## Request Format
```json
{
  "audio": "base64-encoded-mp3-string",
  "language": "en",
  "test_description": "optional description"
}