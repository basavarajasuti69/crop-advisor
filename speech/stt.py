"""Speech-to-text adapter: Transcribe -> Sarvam -> Whisper fallback."""

from __future__ import annotations

import os
from typing import Optional

from apps.api.config import settings  # noqa: E402


class STTAdapter:
    """Transcribe audio bytes to text, falling across providers."""

    def __init__(self) -> None:
        self._transcribe = None
        self._sarvam_key = os.getenv("CA_SARVAM_API_KEY")

    def _transcribe_client(self):
        if self._transcribe is None:
            import boto3
            self._transcribe = boto3.client("transcribe", region_name=settings.region)
        return self._transcribe

    def transcribe(self, audio_bytes: bytes, language: str = "en") -> Optional[str]:
        """Return transcribed text, or None if all providers fail."""
        # 1. Amazon Transcribe (StartTranscribeJob is async; for real-time use
        #    Transcribe Streaming SDK. Here we use a simple synchronous path
        #    via a local file + batch job is too slow, so we fall to Sarvam.)
        #    Keep as placeholder; real implementation uses streaming.
        # 2. Sarvam speech-to-text
        if self._sarvam_key:
            try:
                import httpx
                import base64
                resp = httpx.post(
                    "https://api.sarvam.ai/speech-to-text",
                    headers={"api-subscription-key": self._sarvam_key},
                    json={
                        "audio": base64.b64encode(audio_bytes).decode(),
                        "language_code": language,
                        "model": "bulbul:v1",
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("transcript")
            except Exception:
                pass

        # 3. OpenAI Whisper (local, if installed)
        try:
            import whisper
            model = whisper.load_model("base")
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp = f.name
            result = model.transcribe(tmp, language=language)
            os.unlink(tmp)
            return result.get("text")
        except Exception:
            pass

        return None


stt_adapter = STTAdapter()
