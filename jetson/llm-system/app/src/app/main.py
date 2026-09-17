import base64
import io
import logging
import os
import time
import wave

import numpy as np
import pyaudio
import torch
from openai import OpenAI
from scipy.signal import resample_poly
from silero_vad import load_silero_vad


# ============================================================
# Configuration
# ============================================================

INPUT_RATE = 44100
VAD_RATE = 16000

CHANNELS = 1
FORMAT = pyaudio.paInt16

CHUNK_MS = 32
INPUT_CHUNK = int(INPUT_RATE * CHUNK_MS / 1000)

VAD_CHUNK = 512

VAD_THRESHOLD = 0.5

MIN_SPEECH_MS = 300
MIN_SILENCE_MS = 700

PRE_ROLL_MS = 300

MAX_UTTERANCE_MS = 10000

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    "./recordings",
)

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
# Logging
# ============================================================

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()

logging.basicConfig(
    level=getattr(
        logging,
        LOG_LEVEL,
        logging.INFO,
    ),
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s: "
        "%(message)s"
    ),
    handlers=[
        logging.StreamHandler()
    ],
)

logger = logging.getLogger("audio-app")


# ============================================================
# OpenAI client
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


# ============================================================
# Audio utility
# ============================================================

def find_audio_device(pa):
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)

        logger.info(
            "Audio device %d: name=%s input_channels=%s",
            i,
            info["name"],
            info["maxInputChannels"],
        )

        if (
            info["maxInputChannels"] > 0
            and "USB Microphone" in info["name"]
        ):
            return i

    raise RuntimeError("USB Microphone not found")


def pcm_to_wav_bytes(
    pcm_data: bytes,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

    return buffer.getvalue()


def save_wav(
    wav_data: bytes,
    filename: str,
) -> None:

    try:
        directory = os.path.dirname(filename)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(filename, "wb") as f:
            f.write(wav_data)

    except Exception:
        logger.exception(
            "Failed to save WAV: %s",
            filename,
        )
        raise


def resample_audio(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:

    audio_float = (
        audio.astype(np.float32)
        / 32768.0
    )

    resampled = resample_poly(
        audio_float,
        target_rate,
        source_rate,
    )

    return resampled.astype(
        np.float32
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


# ============================================================
# Save and transcribe
# ============================================================

def process_recording(
    pcm_data: bytes,
    utterance_id: int,
) -> None:

    logger.info(
        "Processing recording #%04d: PCM size=%d bytes",
        utterance_id,
        len(pcm_data),
    )

    # --------------------------------------------------------
    # WAV
    # --------------------------------------------------------

    try:

        wav_data = pcm_to_wav_bytes(
            pcm_data,
            INPUT_RATE,
            CHANNELS,
            2,
        )

        filename = os.path.join(
            OUTPUT_DIR,
            (
                f"utterance_"
                f"{utterance_id:04d}.wav"
            ),
        )

        save_wav(
            wav_data,
            filename,
        )

        logger.info(
            "Saved WAV: %s (%.1f KB)",
            filename,
            len(wav_data) / 1024,
        )

    except Exception:

        logger.exception(
            "Recording processing failed while saving WAV"
        )

        return

    # --------------------------------------------------------
    # Transcription
    # --------------------------------------------------------

    text = transcribe_wav(
        wav_data
    )

    if text:

        logger.info(
            "Transcription: %s",
            text,
        )

    else:

        logger.warning(
            "Transcription failed for recording #%04d",
            utterance_id,
        )


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    logger.info(
        "OpenAI-compatible API configuration:"
    )

    logger.info(
        "  base_url=%s",
        OPENAI_BASE_URL,
    )

    logger.info(
        "  model=%s",
        OPENAI_MODEL,
    )

    logger.info(
        "  output_dir=%s",
        os.path.abspath(OUTPUT_DIR),
    )

    logger.info(
        "Loading Silero VAD"
    )

    try:
        vad_model = load_silero_vad()

    except Exception:

        logger.exception(
            "Failed to load Silero VAD"
        )

        raise

    logger.info(
        "Initializing PyAudio"
    )

    pa = pyaudio.PyAudio()

    logger.info(
        "Audio device count: %d",
        pa.get_device_count(),
    )

    try:
        device_index= find_audio_device(pa)

    except Exception:
        pa.terminate()
        raise

    logger.info(
        "Using input device %d: %s",
        device_index,
        device_info["name"],
    )

    try:

        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=INPUT_CHUNK,
        )

    except Exception:

        logger.exception(
            "Failed to open audio input stream"
        )

        pa.terminate()
        raise

    logger.info(
        "Audio stream opened: "
        "rate=%d chunk=%d",
        INPUT_RATE,
        INPUT_CHUNK,
    )

    logger.info(
        "Listening..."
    )

    speech_started = False

    speech_frames = 0
    silence_frames = 0

    utterance_frames = []

    pre_roll_frames = []

    utterance_id = 1

    min_speech_frames = int(
        MIN_SPEECH_MS / CHUNK_MS
    )

    min_silence_frames = int(
        MIN_SILENCE_MS / CHUNK_MS
    )

    max_utterance_frames = int(
        MAX_UTTERANCE_MS / CHUNK_MS
    )

    pre_roll_frame_count = int(
        PRE_ROLL_MS / CHUNK_MS
    )

    # VAD診断用
    last_vad_log_time = time.monotonic()

    try:

        while True:

            # ------------------------------------------------
            # Audio input
            # ------------------------------------------------

            try:

                data = stream.read(
                    INPUT_CHUNK,
                    exception_on_overflow=False,
                )

            except Exception:

                logger.exception(
                    "Audio input read failed"
                )

                continue

            if not data:

                logger.warning(
                    "Audio input returned empty data"
                )

                continue

            audio = np.frombuffer(
                data,
                dtype=np.int16,
            )

            if len(audio) == 0:

                logger.warning(
                    "Audio input returned zero samples"
                )

                continue

            # ------------------------------------------------
            # 44.1 kHz -> 16 kHz
            # ------------------------------------------------

            vad_audio = resample_audio(
                audio,
                INPUT_RATE,
                VAD_RATE,
            )

            # ------------------------------------------------
            # VAD
            # ------------------------------------------------

            is_speech = False
            max_speech_probability = 0.0

            for start in range(
                0,
                len(vad_audio),
                VAD_CHUNK,
            ):

                chunk = vad_audio[
                    start:start + VAD_CHUNK
                ]

                if len(chunk) < VAD_CHUNK:
                    break

                speech_probability = vad_model(
                    torch.from_numpy(chunk),
                    VAD_RATE,
                ).item()

                max_speech_probability = max(
                    max_speech_probability,
                    speech_probability,
                )

                if (
                    speech_probability
                    >= VAD_THRESHOLD
                ):

                    is_speech = True
                    break

            # ------------------------------------------------
            # Periodic VAD diagnostic log
            # ------------------------------------------------

            now = time.monotonic()

            if (
                now - last_vad_log_time
                >= 5.0
            ):

                logger.debug(
                    "VAD status: speech=%s probability=%.3f recording=%s",
                    is_speech,
                    max_speech_probability,
                    speech_started,
                )

                last_vad_log_time = now

            # ------------------------------------------------
            # Pre-roll
            # ------------------------------------------------

            pre_roll_frames.append(data)

            if (
                len(pre_roll_frames)
                > pre_roll_frame_count
            ):

                pre_roll_frames.pop(0)

            # ------------------------------------------------
            # Speech
            # ------------------------------------------------

            if is_speech:

                speech_frames += 1
                silence_frames = 0

                if not speech_started:

                    if (
                        speech_frames
                        >= min_speech_frames
                    ):

                        speech_started = True

                        utterance_frames = (
                            pre_roll_frames.copy()
                        )

                        logger.info(
                            "Recording started"
                        )

                else:

                    utterance_frames.append(
                        data
                    )

            # ------------------------------------------------
            # Silence
            # ------------------------------------------------

            else:

                silence_frames += 1

                if speech_started:

                    utterance_frames.append(
                        data
                    )

                    if (
                        silence_frames
                        >= min_silence_frames
                    ):

                        logger.info(
                            "Recording ended"
                        )

                        pcm_data = b"".join(
                            utterance_frames
                        )

                        process_recording(
                            pcm_data,
                            utterance_id,
                        )

                        utterance_id += 1

                        # ------------------------------------
                        # Reset
                        # ------------------------------------

                        speech_started = False
                        speech_frames = 0
                        silence_frames = 0
                        utterance_frames = []

            # ------------------------------------------------
            # Maximum utterance length
            # ------------------------------------------------

            if (
                speech_started
                and len(utterance_frames)
                >= max_utterance_frames
            ):

                logger.warning(
                    "Maximum utterance length reached"
                )

                logger.info(
                    "Recording ended"
                )

                pcm_data = b"".join(
                    utterance_frames
                )

                process_recording(
                    pcm_data,
                    utterance_id,
                )

                utterance_id += 1

                speech_started = False
                speech_frames = 0
                silence_frames = 0
                utterance_frames = []

    except KeyboardInterrupt:

        logger.info(
            "Stopping..."
        )

    finally:

        stream.stop_stream()
        stream.close()

        pa.terminate()

        logger.info(
            "Audio application stopped"
        )


if __name__ == "__main__":
    main()
