# Agent Voice Bridge

Real-time voice calls with AI using Twilio Voice + Gemini Live API.

```
Phone ←→ Twilio ←→ Voice Bridge ←→ Gemini Live API
```

## Features

- 📞 Answer phone calls with AI
- 🎤 Real-time voice conversation (low latency)
- 🔄 Interruption handling (barge-in)
- 🔌 Pluggable AI backend (Gemini, OpenAI)

## Quick Start

```bash
# Install
uv sync

# Configure
cp config.yaml.example config.yaml
# Edit config.yaml with your credentials

# Run
uv run voice-bridge
```

## Requirements

- Twilio account with phone number
- Gemini API key (or OpenAI for alternative backend)
- Public URL for Twilio webhook (use ngrok for local dev)

## Configuration

Set environment variables or edit `config.yaml`:

```bash
export TWILIO_ACCOUNT_SID=your_sid
export TWILIO_AUTH_TOKEN=your_token
export GEMINI_API_KEY=your_key
```

## Architecture

See [specs/architecture.md](specs/architecture.md) for detailed design.

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint
uv run ruff check src/
```

## License

MIT
