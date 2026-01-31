"""Audio format conversion utilities.

Twilio sends/receives μ-law encoded audio at 8kHz.
Gemini expects/produces PCM16 at 16kHz or 24kHz.
"""

import base64
import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


# μ-law decoding table (ITU-T G.711)
ULAW_DECODE_TABLE = np.array(
    [
        -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
        -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
        -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
        -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
        -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
        -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
        -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
        -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
        -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
        -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
        -876, -844, -812, -780, -748, -716, -684, -652,
        -620, -588, -556, -524, -492, -460, -428, -396,
        -372, -356, -340, -324, -308, -292, -276, -260,
        -244, -228, -212, -196, -180, -164, -148, -132,
        -120, -112, -104, -96, -88, -80, -72, -64,
        -56, -48, -40, -32, -24, -16, -8, 0,
        32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
        23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
        15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
        11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
        7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
        5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
        3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
        2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
        1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
        1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
        876, 844, 812, 780, 748, 716, 684, 652,
        620, 588, 556, 524, 492, 460, 428, 396,
        372, 356, 340, 324, 308, 292, 276, 260,
        244, 228, 212, 196, 180, 164, 148, 132,
        120, 112, 104, 96, 88, 80, 72, 64,
        56, 48, 40, 32, 24, 16, 8, 0,
    ],
    dtype=np.int16,
)


def ulaw_to_pcm16(ulaw_bytes: bytes) -> np.ndarray:
    """Convert μ-law bytes to PCM16 numpy array.
    
    Args:
        ulaw_bytes: Raw μ-law encoded bytes
        
    Returns:
        PCM16 samples as int16 numpy array
    """
    ulaw_array = np.frombuffer(ulaw_bytes, dtype=np.uint8)
    return ULAW_DECODE_TABLE[ulaw_array]


def pcm16_to_ulaw(pcm16: np.ndarray) -> bytes:
    """Convert PCM16 numpy array to μ-law bytes.
    
    Args:
        pcm16: PCM16 samples as int16 numpy array
        
    Returns:
        μ-law encoded bytes
    """
    # Ensure int16
    pcm16 = pcm16.astype(np.int16)
    
    # μ-law encoding
    sign = (pcm16 < 0).astype(np.uint8) * 0x80
    pcm16 = np.abs(pcm16)
    
    # Bias
    pcm16 = np.clip(pcm16 + 132, 0, 32767)
    
    # Find segment
    exponent = np.floor(np.log2(pcm16.astype(np.float32) + 1)).astype(np.uint8)
    exponent = np.clip(exponent - 7, 0, 7)
    
    # Calculate mantissa
    mantissa = (pcm16 >> (exponent + 3)) & 0x0F
    
    # Combine
    ulaw = ~(sign | (exponent << 4) | mantissa) & 0xFF
    
    return ulaw.astype(np.uint8).tobytes()


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio using linear interpolation.
    
    Args:
        audio: Input audio samples
        from_rate: Source sample rate
        to_rate: Target sample rate
        
    Returns:
        Resampled audio
    """
    if from_rate == to_rate:
        return audio
        
    # Calculate new length
    new_length = int(len(audio) * to_rate / from_rate)
    
    # Linear interpolation
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, new_length)
    
    return np.interp(x_new, x_old, audio).astype(audio.dtype)


def twilio_audio_to_pcm16(payload: str, target_rate: int = 16000) -> np.ndarray:
    """Convert Twilio base64 μ-law audio to PCM16.
    
    Args:
        payload: Base64-encoded μ-law audio from Twilio
        target_rate: Target sample rate (default 16kHz for Gemini)
        
    Returns:
        PCM16 samples at target rate
    """
    # Decode base64
    ulaw_bytes = base64.b64decode(payload)
    
    # Convert to PCM16
    pcm16 = ulaw_to_pcm16(ulaw_bytes)
    
    # Resample from 8kHz to target
    if target_rate != 8000:
        pcm16 = resample(pcm16, 8000, target_rate)
    
    return pcm16


def pcm16_to_twilio_audio(pcm16: np.ndarray, source_rate: int = 24000) -> str:
    """Convert PCM16 to Twilio-compatible base64 μ-law.
    
    Args:
        pcm16: PCM16 samples
        source_rate: Source sample rate (default 24kHz from Gemini)
        
    Returns:
        Base64-encoded μ-law audio for Twilio
    """
    # Resample to 8kHz
    if source_rate != 8000:
        pcm16 = resample(pcm16, source_rate, 8000)
    
    # Convert to μ-law
    ulaw_bytes = pcm16_to_ulaw(pcm16)
    
    # Base64 encode
    return base64.b64encode(ulaw_bytes).decode("ascii")
