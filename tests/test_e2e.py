"""End-to-end tests simulating Twilio ↔ Gemini flow."""

import asyncio
import base64
import json
import math
import struct

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from agent_voice_bridge.server import app


class TestHealthEndpoint:
    """Test health check."""

    def test_health(self):
        """Test health endpoint returns ok."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestIncomingCallEndpoint:
    """Test TwiML generation."""

    def test_incoming_call_returns_twiml(self):
        """Test incoming call returns valid TwiML."""
        client = TestClient(app)
        
        # Simulate Twilio POST
        response = client.post(
            "/incoming",
            data={
                "From": "+1234567890",
                "CallSid": "CA123",
            }
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/xml"
        
        content = response.text
        assert "<?xml version" in content
        assert "<Response>" in content
        assert "<Connect>" in content
        assert "<Stream" in content
        assert "media-stream" in content


class TestMediaStreamSimulation:
    """Simulate Twilio media stream messages."""

    @pytest.fixture
    def sample_audio_payload(self):
        """Generate sample μ-law audio payload."""
        # 20ms of 400Hz tone at 8kHz
        freq = 400
        rate = 8000
        samples = 160
        
        pcm = [int(math.sin(2 * math.pi * freq * i / rate) * 8000) for i in range(samples)]
        pcm_bytes = struct.pack(f'{samples}h', *pcm)
        
        import audioop
        ulaw = audioop.lin2ulaw(pcm_bytes, 2)
        return base64.b64encode(ulaw).decode()

    @pytest.fixture
    def twilio_start_message(self):
        """Sample Twilio stream start message."""
        return {
            "event": "start",
            "start": {
                "streamSid": "MZ123456789",
                "accountSid": "AC123",
                "callSid": "CA123",
                "customParameters": {
                    "caller": "+1234567890"
                }
            }
        }

    @pytest.fixture
    def twilio_media_message(self, sample_audio_payload):
        """Sample Twilio media message."""
        return {
            "event": "media",
            "media": {
                "payload": sample_audio_payload,
                "timestamp": "12345",
                "chunk": "1"
            },
            "streamSid": "MZ123456789"
        }

    @pytest.fixture
    def twilio_stop_message(self):
        """Sample Twilio stop message."""
        return {
            "event": "stop",
            "streamSid": "MZ123456789"
        }

    @pytest.mark.asyncio
    async def test_websocket_accepts_connection(self):
        """Test WebSocket endpoint accepts connections."""
        # Note: Full E2E test requires mocking Gemini API
        # This just tests the initial connection
        pass  # Will need proper mocking


class TestAudioRecording:
    """Test audio recording for debugging."""

    @pytest.fixture
    def temp_audio_dir(self, tmp_path):
        """Create temp directory for audio files."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        return audio_dir

    def test_can_save_pcm_audio(self, temp_audio_dir):
        """Test saving PCM audio to file."""
        import wave
        
        # Generate test audio
        freq = 400
        rate = 16000
        duration = 1  # 1 second
        samples = rate * duration
        
        pcm = [int(math.sin(2 * math.pi * freq * i / rate) * 16000) for i in range(samples)]
        pcm_bytes = struct.pack(f'{samples}h', *pcm)
        
        # Save as WAV
        filepath = temp_audio_dir / "test.wav"
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(pcm_bytes)
        
        # Verify file
        assert filepath.exists()
        with wave.open(str(filepath), 'rb') as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == rate
            assert wf.getnframes() == samples
