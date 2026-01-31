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
        model: str = "gemini-2.0-flash-exp",
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
        self._session: types.AsyncSession | None = None
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
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
        
        # Connect
        self._session = await self._client.aio.live.connect(
            model=self.model,
            config=config,
        ).__aenter__()
        
        self._connected = True
        logger.info("Connected to Gemini Live")

    async def send_audio(self, audio_bytes: bytes):
        """Send audio to Gemini.
        
        Args:
            audio_bytes: PCM16 audio at 16kHz
        """
        if not self._session:
            logger.warning("Cannot send audio - not connected")
            return
            
        # Create audio blob
        audio_blob = types.Blob(
            mime_type="audio/pcm;rate=16000",
            data=audio_bytes,
        )
        
        # Send to Gemini
        await self._session.send(
            input=types.LiveClientRealtimeInput(
                media_chunks=[audio_blob],
            ),
            end_of_turn=False,
        )

    async def receive_audio(self) -> AsyncIterator[bytes]:
        """Receive audio responses from Gemini.
        
        Yields:
            PCM16 audio chunks at 24kHz
        """
        if not self._session:
            return
            
        async for response in self._session.receive():
            # Check for audio data
            if response.data:
                yield response.data
                
            # Check for server content (model responses)
            if response.server_content:
                for part in response.server_content.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        yield part.inline_data.data

    async def interrupt(self):
        """Interrupt the current response (barge-in)."""
        if not self._session:
            return
            
        # Send interrupt signal
        # Note: Gemini automatically handles interruption when new audio is sent
        logger.info("Interrupting Gemini response")

    async def close(self):
        """Close the connection."""
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            self._session = None
            
        self._connected = False
        logger.info("Disconnected from Gemini Live")
