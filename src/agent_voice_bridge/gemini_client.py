"""Gemini Live API client for real-time voice."""

import asyncio
import logging
from typing import AsyncIterator

from google import genai
from google.genai import types

logger = logging.getLogger("voice-bridge.gemini")


class GeminiLiveClient:
    """Client for Gemini Live API real-time voice conversations."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-native-audio-latest",
        system_prompt: str = "",
    ):
        """Initialize the Gemini Live client.
        
        Args:
            api_key: Gemini API key
            model: Model to use (must support live/realtime)
            system_prompt: System instructions for the AI
        """
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        
        self._client: genai.Client | None = None
        self._session = None
        self._context_manager = None
        self._connected = False

    async def connect(self):
        """Connect to Gemini Live API."""
        if self._connected:
            return
            
        logger.info(f"Connecting to Gemini Live ({self.model})...")
        
        # Initialize client
        self._client = genai.Client(api_key=self.api_key)
        
        # Configure live session
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Aoede",  # Natural female voice
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=self.system_prompt)]
            ) if self.system_prompt else None,
        )
        
        # Connect - store the context manager and enter it
        self._context_manager = self._client.aio.live.connect(
            model=self.model,
            config=config,
        )
        self._session = await self._context_manager.__aenter__()
        
        self._connected = True
        logger.info("Connected to Gemini Live")

    _audio_chunks_sent: int = 0
    _last_audio_log: float = 0
    
    async def send_audio(self, audio_bytes: bytes):
        """Send audio to Gemini.
        
        Args:
            audio_bytes: PCM16 audio at 16kHz
        """
        import struct
        import time
        
        if not self._session:
            logger.warning("Cannot send audio - not connected")
            return
        
        # Calculate audio level for debugging
        if len(audio_bytes) >= 2:
            samples = struct.unpack(f'{len(audio_bytes)//2}h', audio_bytes)
            max_level = max(abs(s) for s in samples)
            avg_level = sum(abs(s) for s in samples) // len(samples)
            
            self._audio_chunks_sent += 1
            now = time.time()
            if now - self._last_audio_log > 2.0:  # Log every 2 seconds
                logger.info(f"🎤 Audio stats: chunks={self._audio_chunks_sent}, max={max_level}, avg={avg_level}")
                self._last_audio_log = now
            
        try:
            # Send realtime audio input
            await self._session.send_realtime_input(
                audio=types.Blob(
                    mime_type="audio/pcm;rate=16000",
                    data=audio_bytes,
                )
            )
        except Exception as e:
            logger.error(f"Error sending audio: {e}")

    async def receive_audio(self) -> AsyncIterator[bytes]:
        """Receive audio responses from Gemini.
        
        Yields:
            PCM16 audio chunks at 24kHz
        """
        if not self._session:
            logger.warning("receive_audio called but no session")
            return
        
        logger.info("Starting to receive audio from Gemini...")
        
        try:
            async for response in self._session.receive():
                resp_type = type(response).__name__
                logger.info(f"📩 Gemini response: {resp_type}")
                
                # Check for audio data in different response formats
                if hasattr(response, 'data') and response.data:
                    logger.info(f"🔊 Got direct data: {len(response.data)} bytes")
                    yield response.data
                    
                # Check for server content (model responses)
                if hasattr(response, 'server_content') and response.server_content:
                    sc = response.server_content
                    logger.debug(f"Server content: turn_complete={getattr(sc, 'turn_complete', None)}")
                    
                    if hasattr(sc, 'model_turn') and sc.model_turn:
                        for part in sc.model_turn.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                if part.inline_data.data:
                                    logger.info(f"🔊 Got inline audio: {len(part.inline_data.data)} bytes")
                                    yield part.inline_data.data
                            elif hasattr(part, 'text') and part.text:
                                logger.info(f"📝 Got text: {part.text[:100]}...")
                                
                # Check for tool calls or other content
                if hasattr(response, 'tool_call'):
                    logger.info(f"🔧 Got tool call: {response.tool_call}")
                    
        except asyncio.CancelledError:
            logger.info("receive_audio cancelled")
            raise
        except Exception as e:
            logger.error(f"Error receiving audio: {e}", exc_info=True)

    async def close(self):
        """Close the connection."""
        if self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            self._context_manager = None
            self._session = None
            
        self._connected = False
        logger.info("Disconnected from Gemini Live")
