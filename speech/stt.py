"""Speech-to-text adapter: Transcribe -> Sarvam -> Whisper fallback."""

from __future__ import annotations

import os
from typing import Optional

from apps.api.config import settings  # noqa: E402


class STTAdapter:
    """Transcribe audio bytes to text, falling across providers."""

    # Language -> Sarvam BCP-47 language_code. The API rejects the bare
    # "hi" / "kn" codes.
    SARVAM_LANGUAGES = {
        "en": "en-IN",
        "hi": "hi-IN",
        "kn": "kn-IN",
    }

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
                # The endpoint takes multipart/form-data with the raw audio,
                # not a JSON body with base64.
                fields = {"model": "saaras:v3"}
                language_code = self.SARVAM_LANGUAGES.get(language)
                if language_code:
                    fields["language_code"] = language_code
                resp = httpx.post(
                    "https://api.sarvam.ai/speech-to-text",
                    headers={"api-subscription-key": self._sarvam_key},
                    data=fields,
                    files={"file": ("audio.webm", audio_bytes, "application/octet-stream")},
                    timeout=30,
                )
                if resp.status_code == 200:
                    return resp.json().get("transcript")
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
