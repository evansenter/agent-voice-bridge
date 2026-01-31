"""FastAPI server for voice bridge - Twilio + Gemini Live API."""

import asyncio
import audioop
import base64
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from google import genai
from google.genai import types

from agent_voice_bridge.config import Settings, get_settings

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
    description="Real-time voice calls with AI using Twilio + Gemini Live",
    version="0.2.0",
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
    
    form = await request.form()
    caller = form.get("From", "unknown")
    call_sid = form.get("CallSid", "unknown")
    logger.info(f"📞 INCOMING CALL from {caller} (CallSid: {call_sid})")
    
    # Build WebSocket URL
    ws_url = settings.public_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/media-stream"
    logger.info(f"📡 WebSocket URL: {ws_url}")
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Connecting to Gemini.</Say>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="caller" value="{caller}" />
        </Stream>
    </Connect>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")


# --- Audio Processing (using audioop for reliability) ---

def process_incoming_audio(chunk: bytes, state) -> tuple[bytes, any]:
    """Convert Twilio μ-law 8kHz to PCM 16kHz for Gemini."""
    # μ-law to PCM at 8kHz
    pcm_8k = audioop.ulaw2lin(chunk, 2)
    # Upsample 8kHz to 16kHz
    pcm_16k, new_state = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, state)
    return pcm_16k, new_state


def process_outgoing_audio(audio_data: bytes, state) -> tuple[str, any]:
    """Convert Gemini PCM 24kHz to Twilio μ-law 8kHz base64."""
    # Downsample 24kHz to 8kHz
    pcm_8k, new_state = audioop.ratecv(audio_data, 2, 1, 24000, 8000, state)
    # PCM to μ-law
    mulaw = audioop.lin2ulaw(pcm_8k, 2)
    # Base64 encode
    return base64.b64encode(mulaw).decode("utf-8"), new_state


# --- WebSocket Handler ---

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handle Twilio media stream WebSocket."""
    await websocket.accept()
    logger.info("📱 Twilio WebSocket connected")
    
    settings = get_settings()
    stream_sid: str | None = None
    
    # Initialize Gemini client
    client = genai.Client(api_key=settings.gemini_api_key)
    
    config = types.LiveConnectConfig(
        system_instruction=settings.system_prompt,
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Aoede"
                )
            )
        )
    )
    
    logger.info(f"🤖 Connecting to Gemini ({settings.gemini_model})...")
    
    async with client.aio.live.connect(
        model=settings.gemini_model,
        config=config,
    ) as session:
        logger.info("✅ Connected to Gemini")
        
        # Shared state
        stream_info = {"sid": None}
        downsample_state = None
        
        async def send_to_twilio(b64_audio: str):
            """Send audio back to Twilio."""
            nonlocal downsample_state
            if not stream_info["sid"]:
                return
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_info["sid"],
                "media": {"payload": b64_audio},
            })
        
        async def gemini_receiver():
            """Receive audio from Gemini and send to Twilio."""
            nonlocal downsample_state
            chunks_sent = 0
            try:
                async for response in session.receive():
                    if response.server_content is None:
                        continue
                    
                    model_turn = response.server_content.model_turn
                    if model_turn and model_turn.parts:
                        for part in model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                audio_data = part.inline_data.data
                                b64_audio, downsample_state = process_outgoing_audio(
                                    audio_data, downsample_state
                                )
                                await send_to_twilio(b64_audio)
                                chunks_sent += 1
                                if chunks_sent % 20 == 1:
                                    logger.info(f"📤 Sent {chunks_sent} audio chunks to Twilio")
                    
                    if response.server_content.turn_complete:
                        logger.info("🔄 Gemini turn complete")
                        
            except asyncio.CancelledError:
                logger.info("Gemini receiver cancelled")
            except Exception as e:
                logger.error(f"Gemini receiver error: {e}")
        
        # Start receiver task
        receive_task = asyncio.create_task(gemini_receiver())
        
        # Audio buffer (send every ~300ms = 9600 bytes at 16kHz)
        audio_buffer = bytearray()
        upsample_state = None
        BUFFER_SIZE = 9600  # ~300ms at 16kHz, 16-bit
        
        try:
            async for message in websocket.iter_text():
                data = json.loads(message)
                event = data.get("event")
                
                if event == "connected":
                    logger.info("Twilio stream connected")
                    
                elif event == "start":
                    start_data = data.get("start", {})
                    stream_info["sid"] = start_data.get("streamSid")
                    params = start_data.get("customParameters", {})
                    caller = params.get("caller", "unknown")
                    logger.info(f"📞 Stream started: {stream_info['sid'][:20]}... from {caller}")
                    
                elif event == "media":
                    payload = data.get("media", {}).get("payload", "")
                    if payload:
                        chunk = base64.b64decode(payload)
                        pcm_16k, upsample_state = process_incoming_audio(chunk, upsample_state)
                        audio_buffer.extend(pcm_16k)
                        
                        # Send buffered audio to Gemini
                        if len(audio_buffer) >= BUFFER_SIZE:
                            await session.send(
                                input={
                                    "data": bytes(audio_buffer),
                                    "mime_type": "audio/pcm;rate=16000"
                                },
                                end_of_turn=False
                            )
                            audio_buffer.clear()
                            
                elif event == "stop":
                    logger.info("Stream stopped")
                    break
                    
        except WebSocketDisconnect:
            logger.info("Twilio WebSocket disconnected")
        except Exception as e:
            logger.error(f"Media stream error: {e}")
        finally:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
            logger.info("Media stream closed")


def main():
    """Run the server."""
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "agent_voice_bridge.server:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
