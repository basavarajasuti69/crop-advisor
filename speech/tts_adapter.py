"""Text-to-speech adapter with provider fallback: Polly -> Sarvam -> Google."""

from __future__ import annotations

import os
from typing import Optional

from apps.api.config import settings  # noqa: E402


class TTSAdapter:
    """Speak text in the farmer's language, falling across providers."""

    # Language -> Polly voice mapping (neural voices where available).
    # Polly has no Kannada voice, so "kn" is deliberately absent: sending
    # Kannada text to Polly would read it out with an English voice.
    POLLY_VOICES = {
        "en": "Joanna",
        "hi": "Aditi",
    }

    # Language -> Sarvam target_language_code (Indian-language provider).
    SARVAM_LANGUAGES = {
        "en": "en-IN",
        "hi": "hi-IN",
        "kn": "kn-IN",
    }

    # Language -> Sarvam speaker. The API only accepts documented speaker
    # names and "meera" is not one of them, so every call was rejected.
    # shubh and ishita cover all 11 Sarvam languages, Kannada included.
    SARVAM_SPEAKERS = {
        "en": "shubh",
        "hi": "ishita",
        "kn": "ishita",
    }

    # Language -> Google Cloud TTS language_code.
    GOOGLE_LANGUAGES = {
        "en": "en-IN",
        "hi": "hi-IN",
        "kn": "kn-IN",
    }

    def __init__(self) -> None:
        self._polly = None
        self._sarvam_key = os.getenv("CA_SARVAM_API_KEY")
        self._google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    def _polly_client(self):
        if self._polly is None:
            import boto3
            self._polly = boto3.client("polly", region=settings.region)
        return self._polly

    def configured_providers(self, language: str = "en") -> list[str]:
        """Speech providers that can serve this language and have credentials."""
        providers = []
        if language in self.POLLY_VOICES:
            providers.append("aws_polly")
        if self._sarvam_key and language in self.SARVAM_LANGUAGES:
            providers.append("sarvam")
        if self._google_creds and language in self.GOOGLE_LANGUAGES:
            providers.append("google")
        return providers

    def synthesize(self, text: str, language: str = "en") -> Optional[bytes]:
        """Return MP3 audio bytes, or None if all providers fail."""
        # 1. Amazon Polly (skipped for languages Polly cannot speak)
        voice = self.POLLY_VOICES.get(language)
        if voice:
            try:
                client = self._polly_client()
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
        target_language = self.SARVAM_LANGUAGES.get(language)
        speaker = self.SARVAM_SPEAKERS.get(language)
        if self._sarvam_key and target_language and speaker:
            try:
                import httpx
                resp = httpx.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers={"api-subscription-key": self._sarvam_key},
                    json={
                        "inputs": [text],
                        "target_language_code": target_language,
                        "speaker": speaker,
                        "pace": 1.0,
                        "speech_sample_rate": 22050,
                        "enable_preprocessing": True,
                        # Only bulbul:v2, bulbul:v3 and bulbul:v4-flash exist.
                        "model": "bulbul:v3",
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
        google_language = self.GOOGLE_LANGUAGES.get(language)
        if self._google_creds and google_language:
            try:
                from google.cloud import texttospeech
                client = texttospeech.TextToSpeechClient()
                synthesis_input = texttospeech.SynthesisInput(text=text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code=google_language,
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
