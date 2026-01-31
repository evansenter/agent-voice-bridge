"""Tests for audio processing functions."""

import audioop
import base64
import math
import struct

import pytest

from agent_voice_bridge.server import process_incoming_audio, process_outgoing_audio


class TestIncomingAudio:
    """Tests for Twilio → Gemini audio conversion."""

    def test_ulaw_to_pcm_basic(self):
        """Test μ-law to PCM conversion."""
        # μ-law silence (0xFF) should produce near-zero PCM
        silence = bytes([0xFF] * 160)  # 20ms at 8kHz
        pcm, state = process_incoming_audio(silence, None)
        
        # Output should be ~2x samples (16kHz from 8kHz), 16-bit
        # Rate conversion may have slight rounding
        assert 630 <= len(pcm) <= 650  # ~160 * 2 * 2
        
        # Check silence produces low amplitude
        samples = struct.unpack(f'{len(pcm)//2}h', pcm)
        max_amp = max(abs(s) for s in samples)
        assert max_amp < 100  # Near silence

    def test_ulaw_to_pcm_signal(self):
        """Test μ-law decoding preserves signal."""
        # Generate a simple tone in μ-law
        # First create PCM sine wave
        freq = 400  # Hz
        rate = 8000
        duration = 0.02  # 20ms
        samples = int(rate * duration)
        
        pcm_samples = [int(math.sin(2 * math.pi * freq * i / rate) * 8000) for i in range(samples)]
        pcm_bytes = struct.pack(f'{samples}h', *pcm_samples)
        
        # Convert to μ-law
        ulaw = audioop.lin2ulaw(pcm_bytes, 2)
        
        # Now convert back through our function
        pcm_out, state = process_incoming_audio(ulaw, None)
        
        # Should have signal
        out_samples = struct.unpack(f'{len(pcm_out)//2}h', pcm_out)
        max_amp = max(abs(s) for s in out_samples)
        assert max_amp > 1000  # Significant signal

    def test_ulaw_state_continuity(self):
        """Test rate conversion state is maintained across chunks."""
        # Send two consecutive chunks
        chunk1 = bytes([0x80] * 160)  # Some non-silence
        chunk2 = bytes([0x80] * 160)
        
        pcm1, state1 = process_incoming_audio(chunk1, None)
        pcm2, state2 = process_incoming_audio(chunk2, state1)
        
        # State should be updated
        assert state1 is not None
        assert state2 is not None


class TestOutgoingAudio:
    """Tests for Gemini → Twilio audio conversion."""

    def test_pcm_to_ulaw_basic(self):
        """Test PCM to μ-law conversion."""
        # Generate 24kHz PCM silence
        silence = bytes([0] * 480)  # 10ms at 24kHz, 16-bit
        
        b64_out, state = process_outgoing_audio(silence, None)
        
        # Should be base64 encoded
        assert b64_out  # Not empty
        decoded = base64.b64decode(b64_out)
        assert len(decoded) > 0

    def test_pcm_to_ulaw_signal(self):
        """Test PCM to μ-law preserves signal."""
        # Generate 24kHz tone
        freq = 400
        rate = 24000
        duration = 0.01  # 10ms
        samples = int(rate * duration)
        
        pcm_samples = [int(math.sin(2 * math.pi * freq * i / rate) * 16000) for i in range(samples)]
        pcm_bytes = struct.pack(f'{samples}h', *pcm_samples)
        
        b64_out, state = process_outgoing_audio(pcm_bytes, None)
        
        # Decode and check
        ulaw = base64.b64decode(b64_out)
        assert len(ulaw) > 0
        
        # Convert back to PCM to verify signal preserved
        pcm_back = audioop.ulaw2lin(ulaw, 2)
        back_samples = struct.unpack(f'{len(pcm_back)//2}h', pcm_back)
        max_amp = max(abs(s) for s in back_samples)
        assert max_amp > 1000  # Signal preserved

    def test_downsample_ratio(self):
        """Test 24kHz → 8kHz downsampling ratio."""
        # 240 samples at 24kHz = 10ms
        pcm_24k = bytes([0] * 480)  # 240 samples * 2 bytes
        
        b64_out, state = process_outgoing_audio(pcm_24k, None)
        ulaw = base64.b64decode(b64_out)
        
        # Should downsample 3:1 (24kHz → 8kHz)
        # 240 / 3 = 80 samples, μ-law is 1 byte per sample
        assert len(ulaw) == 80

    def test_state_continuity(self):
        """Test rate conversion state is maintained."""
        chunk1 = bytes([0] * 480)
        chunk2 = bytes([0] * 480)
        
        b64_1, state1 = process_outgoing_audio(chunk1, None)
        b64_2, state2 = process_outgoing_audio(chunk2, state1)
        
        assert state1 is not None
        assert state2 is not None


class TestRoundTrip:
    """Test audio round-trip conversion."""

    def test_tone_round_trip(self):
        """Test that a tone survives Twilio→Gemini→Twilio."""
        # Generate original 8kHz tone
        freq = 500
        rate = 8000
        samples = 160  # 20ms
        
        pcm_8k = [int(math.sin(2 * math.pi * freq * i / rate) * 8000) for i in range(samples)]
        pcm_bytes = struct.pack(f'{samples}h', *pcm_8k)
        ulaw_in = audioop.lin2ulaw(pcm_bytes, 2)
        
        # Twilio → Gemini (8kHz μ-law → 16kHz PCM)
        pcm_16k, up_state = process_incoming_audio(ulaw_in, None)
        
        # Simulate Gemini processing (pretend it outputs at 24kHz)
        # Just upsample our 16kHz to 24kHz for testing
        pcm_24k, _ = audioop.ratecv(pcm_16k, 2, 1, 16000, 24000, None)
        
        # Gemini → Twilio (24kHz PCM → 8kHz μ-law)
        b64_out, down_state = process_outgoing_audio(pcm_24k, None)
        ulaw_out = base64.b64decode(b64_out)
        
        # Convert back to PCM for analysis
        pcm_final = audioop.ulaw2lin(ulaw_out, 2)
        final_samples = struct.unpack(f'{len(pcm_final)//2}h', pcm_final)
        
        # Check we still have a signal
        max_amp = max(abs(s) for s in final_samples)
        assert max_amp > 500  # Signal survived (some loss expected)
