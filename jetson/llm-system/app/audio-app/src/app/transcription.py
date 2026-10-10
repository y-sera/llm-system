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

# type: input_audio or audio
#   input_audio: OpenAI API Compatible type(e.g. ollama)
#   audio: LiteRT-LM type(at v0.18.0)
#          ref: https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/api/cpp/conversation.md
OPENAI_API_AUDIO_INPUT_TYPE = os.getenv("OPENAI_API_AUDIO_INPUT_TYPE", "input_audio")

match OPENAI_API_AUDIO_INPUT_TYPE:
    case "input_audio":
        logger.info("OpenAI API audio input type: input_audio. OpenAI API compatible.")
    case "audio":
        logger.info("OpenAI API audio input type: audio. LiteRT-LM compatible.")
    case _:
        logger.error("OpenAI API audio input type: %s is unsupported.", OPEN_API_AUDIO_INPUT_TYPE)


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

    if OPENAI_API_AUDIO_INPUT_TYPE == "input_audio":
        audio_input_dict = {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_data,
                    "format": "wav",
                }
        }
    elif OPENAI_API_AUDIO_INPUT_TYPE == "audio":
        audio_input_dict = {
                "type": "audio",
                "blob": audio_data,
        }

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
                        audio_input_dict,
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
