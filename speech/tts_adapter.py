"""Text-to-speech adapter with provider fallback: Polly -> Sarvam -> Google."""

from __future__ import annotations

import os
from typing import Optional

from apps.api.config import settings  # noqa: E402


class TTSAdapter:
    """Speak text in the farmer's language, falling across providers."""

    # Language -> Polly voice mapping (neural voices where available)
    POLLY_VOICES = {
        "en": "Joanna",
        "hi": "Aditi",
    }

    def __init__(self) -> None:
        self._polly = None
        self._sarvam_key = os.getenv("CA_SARVAM_API_KEY")
        self._google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    def _polly_client(self):
        if self._polly is None:
            import boto3
            self._polly = boto3.client("polly", region_name=settings.region)
        return self._polly

    def synthesize(self, text: str, language: str = "en") -> Optional[bytes]:
        """Return MP3 audio bytes, or None if all providers fail."""
        # 1. Amazon Polly
        try:
            client = self._polly_client()
            voice = self.POLLY_VOICES.get(language, "Joanna")
            resp = client.synthesize_speech(
                Text=text,
                OutputFormat="mp3",
                VoiceId=voice,
                Engine="neural",
            )
            return resp["AudioStream"].read()
        except Exception:
            pass

        # 2. Sarvam (Indian-language focused)
        if self._sarvam_key:
            try:
                import httpx
                resp = httpx.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers={"api-subscription-key": self._sarvam_key},
                    json={
                        "inputs": [text],
                        "target_language_code": language,
                        "speaker": "meera",
                        "pitch": 0,
                        "pace": 1.0,
                        "loudness": 1.0,
                        "speech_sample_rate": 22050,
                        "enable_preprocessing": True,
                        "model": "bulbul:v1",
                    },
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    import base64
                    return base64.b64decode(data["audios"][0])
            except Exception:
                pass

        # 3. Google Cloud TTS
        if self._google_creds:
            try:
                from google.cloud import texttospeech
                client = texttospeech.TextToSpeechClient()
                synthesis_input = texttospeech.SynthesisInput(text=text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code=language,
                    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                resp = client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                return resp.audio_content
            except Exception:
                pass

        return None


tts_adapter = TTSAdapter()
