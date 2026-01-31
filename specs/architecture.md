# Agent Voice Bridge - Architecture Spec

## Overview

Real-time voice calls with AI using Twilio Voice + Gemini Live API (or OpenAI Realtime).

```
┌─────────────┐     ┌─────────────────┐     ┌────────────────┐     ┌──────────────┐
│   Phone     │◄───►│  Twilio Voice   │◄───►│  Voice Bridge  │◄───►│  Gemini Live │
│   (PSTN)    │     │  (Media Stream) │     │  (FastAPI+WS)  │     │  or OpenAI   │
└─────────────┘     └─────────────────┘     └────────────────┘     └──────────────┘
```

## Components

### 1. Twilio Voice Webhook
- Answers incoming calls
- Returns TwiML to start media stream
- Connects to our WebSocket endpoint

### 2. Voice Bridge Server (FastAPI)
- `/incoming` - Twilio webhook, returns TwiML
- `/media-stream` - WebSocket for Twilio media
- Handles audio format conversion (μ-law ↔ PCM)
- Manages Gemini/OpenAI session lifecycle

### 3. AI Backend (pluggable)
- **Gemini Live API** - Google's real-time voice
- **OpenAI Realtime** - Alternative backend
- Future: Claude voice when available

## Audio Flow

### Inbound (User → AI)
1. User speaks into phone
2. Twilio captures audio, encodes as μ-law 8kHz
3. Twilio streams base64-encoded audio chunks via WebSocket
4. Bridge decodes μ-law → PCM 16-bit
5. Bridge resamples 8kHz → 16kHz (if needed for AI)
6. Bridge streams to Gemini Live API

### Outbound (AI → User)
1. Gemini generates audio response (PCM 24kHz)
2. Bridge resamples 24kHz → 8kHz
3. Bridge encodes PCM → μ-law
4. Bridge sends base64 audio to Twilio WebSocket
5. Twilio plays audio to caller

## Interruption Handling (Barge-in)

Critical for natural conversation:
1. Detect user speech during AI playback
2. Immediately stop AI audio generation
3. Buffer and process user's interruption
4. Resume with new AI response

## Configuration

```yaml
# config.yaml
twilio:
  account_sid: ${TWILIO_ACCOUNT_SID}
  auth_token: ${TWILIO_AUTH_TOKEN}
  phone_number: "+1234567890"

ai:
  provider: gemini  # or openai
  gemini:
    api_key: ${GEMINI_API_KEY}
    model: gemini-2.0-flash-live
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o-realtime

server:
  host: 0.0.0.0
  port: 8082
  public_url: https://voice.example.com  # For Twilio webhook
```

## Twilio Setup

1. Buy a phone number (~$1/mo)
2. Configure Voice webhook: `https://your-server/incoming`
3. Enable Media Streams in TwiML

## TwiML Response

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://your-server/media-stream">
      <Parameter name="caller" value="{{From}}" />
    </Stream>
  </Connect>
</Response>
```

## Project Structure

```
agent-voice-bridge/
├── src/
│   └── agent_voice_bridge/
│       ├── __init__.py
│       ├── server.py          # FastAPI app
│       ├── twilio_handler.py  # Webhook + media stream
│       ├── audio.py           # Format conversion
│       ├── gemini_client.py   # Gemini Live integration
│       └── openai_client.py   # OpenAI Realtime integration
├── tests/
├── pyproject.toml
├── config.yaml.example
└── README.md
```

## Dependencies

- `fastapi` + `uvicorn` - Web server
- `websockets` - WebSocket handling
- `audioop-lts` - μ-law encoding (Python 3.13+ compatible)
- `numpy` - Audio resampling
- `google-genai` - Gemini API
- `openai` - OpenAI API (optional)

## Deployment

Options:
1. **speck-vm** - Run alongside other agent services
2. **Cloud Run** - Serverless, scales with calls
3. **Fly.io** - Low latency, global edge

Needs stable WebSocket connection, so serverless may have cold start issues.

## Security

- Validate Twilio request signatures
- Rate limit by caller ID
- Optional: Allowlist of phone numbers
- Secure WebSocket (wss://)

## Future Enhancements

1. **Context injection** - Pass caller info to AI
2. **Tool calling** - AI can trigger actions mid-call
3. **Call recording** - Store transcripts
4. **Multi-turn memory** - Remember previous calls
5. **Voicemail** - AI takes messages when unavailable
6. **Outbound calls** - AI initiates calls (reminders, alerts)
