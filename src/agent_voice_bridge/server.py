"""FastAPI server for voice bridge."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from agent_voice_bridge.audio import pcm16_to_twilio_audio, twilio_audio_to_pcm16
from agent_voice_bridge.config import Settings, get_settings
from agent_voice_bridge.gemini_client import GeminiLiveClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voice-bridge")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Voice Bridge starting...")
    yield
    logger.info("Voice Bridge shutting down...")


app = FastAPI(
    title="Agent Voice Bridge",
    description="Real-time voice calls with AI",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/incoming")
async def incoming_call(request: Request):
    """Handle incoming Twilio call - return TwiML to start media stream."""
    settings = get_settings()
    
    # Get caller info from form data
    form = await request.form()
    caller = form.get("From", "unknown")
    logger.info(f"Incoming call from {caller}")
    
    # Build WebSocket URL
    ws_url = settings.server.public_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/media-stream"
    
    # Return TwiML
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Connecting you to the AI assistant.</Say>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="caller" value="{caller}" />
        </Stream>
    </Connect>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handle Twilio media stream WebSocket."""
    await websocket.accept()
    logger.info("Media stream connected")
    
    settings = get_settings()
    stream_sid: str | None = None
    caller: str = "unknown"
    
    # Initialize Gemini client
    gemini_client: GeminiLiveClient | None = None
    gemini_task: asyncio.Task | None = None
    
    async def send_audio_to_twilio(audio_bytes: bytes):
        """Send audio back to Twilio."""
        if stream_sid is None:
            return
            
        # Convert PCM16 to Twilio format
        import numpy as np
        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
        payload = pcm16_to_twilio_audio(pcm16, source_rate=24000)
        
        # Send to Twilio
        await websocket.send_json({
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": payload,
            },
        })
    
    async def handle_gemini_responses():
        """Handle responses from Gemini."""
        if gemini_client is None:
            return
            
        try:
            async for audio_chunk in gemini_client.receive_audio():
                await send_audio_to_twilio(audio_chunk)
        except Exception as e:
            logger.error(f"Gemini response error: {e}")
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get("event")
            
            if event == "connected":
                logger.info("Twilio stream connected")
                
            elif event == "start":
                # Stream started - get metadata
                start_data = data.get("start", {})
                stream_sid = start_data.get("streamSid")
                
                # Get custom parameters
                params = start_data.get("customParameters", {})
                caller = params.get("caller", "unknown")
                
                logger.info(f"Stream started: {stream_sid} from {caller}")
                
                # Initialize Gemini session
                gemini_client = GeminiLiveClient(
                    api_key=settings.gemini.api_key,
                    model=settings.gemini.model,
                    system_prompt=settings.system_prompt,
                )
                await gemini_client.connect()
                
                # Start receiving responses
                gemini_task = asyncio.create_task(handle_gemini_responses())
                
            elif event == "media":
                # Audio data from caller
                media = data.get("media", {})
                payload = media.get("payload", "")
                
                if payload and gemini_client:
                    # Convert to PCM and send to Gemini
                    pcm16 = twilio_audio_to_pcm16(payload, target_rate=16000)
                    await gemini_client.send_audio(pcm16.tobytes())
                    
            elif event == "stop":
                logger.info("Stream stopped")
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Media stream error: {e}")
    finally:
        # Cleanup
        if gemini_task:
            gemini_task.cancel()
        if gemini_client:
            await gemini_client.close()
        logger.info("Media stream closed")


def main():
    """Run the server."""
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "agent_voice_bridge.server:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
