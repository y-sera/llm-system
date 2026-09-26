import base64
import os

from openai import OpenAI

from .logging_config import logger


# ============================================================
# OpenAI configuration
# ============================================================

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
OPENAI_MODEL = os.environ["OPENAI_MODEL"]

OPENAI_PROMPT = os.getenv(
    "OPENAI_PROMPT",
    (
        "この音声を日本語で"
        "文字起こししてください。"
        "音声に含まれている発話だけを"
        "返してください。"
        "説明や補足は不要です。"
    ),
)


# ============================================================
# OpenAI client
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


# ============================================================
# Transcription
# ============================================================

def transcribe_wav(
    wav_data: bytes,
) -> str | None:
    logger.info(
        "Sending audio to OpenAI-compatible API"
    )

    logger.debug(
        "API request: base_url=%s model=%s audio_size=%d bytes",
        OPENAI_BASE_URL,
        OPENAI_MODEL,
        len(wav_data),
    )

    # --------------------------------------------------------
    # WAV -> Base64
    # --------------------------------------------------------

    audio_data = base64.b64encode(
        wav_data
    ).decode("ascii")

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": OPENAI_PROMPT,
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_data,
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
            temperature=0,
        )

    except Exception as e:

        logger.exception(
            "Transcription API request failed: %s",
            e,
        )

        return None

    logger.debug(
        "API response received: %s",
        response,
    )

    try:
        text = response.choices[0].message.content

    except (
        AttributeError,
        IndexError,
        TypeError,
    ):
        logger.error(
            "Unexpected API response: %r",
            response,
        )

        return None

    if not text:
        logger.warning(
            "API returned empty transcription"
        )

        return None

    return text
