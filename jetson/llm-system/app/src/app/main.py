import io
import logging
import os
import wave

import numpy as np
import pyaudio

from scipy.signal import resample_poly
from silero_vad import load_silero_vad


# ============================================================
# Logging
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("audio-app")


# ============================================================
# Audio configuration
# ============================================================

INPUT_RATE = 44100
VAD_RATE = 16000

CHANNELS = 1
FORMAT = pyaudio.paInt16

# PyAudio capture chunk
CHUNK_MS = 32
INPUT_CHUNK = int(INPUT_RATE * CHUNK_MS / 1000)

# Silero VAD uses 512 samples at 16 kHz
VAD_CHUNK = 512


# ============================================================
# VAD configuration
# ============================================================

VAD_THRESHOLD = 0.5

MIN_SPEECH_MS = 300
MIN_SILENCE_MS = 700

PRE_ROLL_MS = 300
POST_ROLL_MS = 200

MAX_UTTERANCE_MS = 10000


# ============================================================
# Load model
# ============================================================

logger.info("Loading Silero VAD...")

vad_model = load_silero_vad()

logger.info("VAD loaded.")


# ============================================================
# Utilities
# ============================================================

def resample_to_16k(audio: np.ndarray) -> np.ndarray:
    """
    int16 mono 44.1kHz -> float32 mono 16kHz
    """

    audio_f32 = audio.astype(np.float32) / 32768.0

    resampled = resample_poly(
        audio_f32,
        VAD_RATE,
        INPUT_RATE,
    )

    return resampled.astype(np.float32)


def save_wav(
    path: str,
    pcm: np.ndarray,
    sample_rate: int,
):
    """
    float32 [-1, 1] -> PCM16 WAV
    """

    pcm16 = np.clip(
        pcm * 32767.0,
        -32768,
        32767,
    ).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


# ============================================================
# Main
# ============================================================

logger.info("Initializing PyAudio...")

audio = pyaudio.PyAudio()

logger.info("PyAudio initialized.")

logger.info("Audio devices:")

device_count = audio.get_device_count()

logger.info("Device count: %d", device_count)

for i in range(device_count):
    info = audio.get_device_info_by_index(i)

    if info["maxInputChannels"] > 0:
        logger.info(
            "  [%d] %s rate=%s input_channels=%s",
            i,
            info["name"],
            info["defaultSampleRate"],
            info["maxInputChannels"],
        )


# Your container currently exposes the USB microphone as index 0.
DEVICE_INDEX = 0

logger.info(
    "Opening input device index=%d...",
    DEVICE_INDEX,
)

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=INPUT_RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=INPUT_CHUNK,
)

logger.info("Audio stream opened successfully.")

logger.info("  sample_rate=%d", INPUT_RATE)
logger.info("  channels=%d", CHANNELS)
logger.info(
    "  chunk=%d samples (%d ms)",
    INPUT_CHUNK,
    CHUNK_MS,
)

logger.info("Listening...")
logger.info("Speak into the microphone.")
logger.info("Press Ctrl+C to stop.")


# ------------------------------------------------------------
# State
# ------------------------------------------------------------

speech_active = False

utterance = []

speech_duration_ms = 0
silence_duration_ms = 0
utterance_duration_ms = 0

utterance_number = 0

# Keep some audio before speech starts.
pre_roll_samples = int(
    VAD_RATE * PRE_ROLL_MS / 1000
)

pre_roll = np.zeros(
    pre_roll_samples,
    dtype=np.float32,
)


# Silero VAD state
vad_state = None


try:

    while True:

        # ----------------------------------------------------
        # Capture audio
        # ----------------------------------------------------

        raw = stream.read(
            INPUT_CHUNK,
            exception_on_overflow=False,
        )

        audio_input = np.frombuffer(
            raw,
            dtype=np.int16,
        )

        # ----------------------------------------------------
        # Resample 44.1k -> 16k
        # ----------------------------------------------------

        audio_16k = resample_to_16k(audio_input)

        # ----------------------------------------------------
        # Feed VAD in 512-sample chunks
        # ----------------------------------------------------

        offset = 0

        while offset + VAD_CHUNK <= len(audio_16k):

            chunk = audio_16k[
                offset:
                offset + VAD_CHUNK
            ]

            offset += VAD_CHUNK

            # Silero VAD expects a torch tensor
            import torch

            tensor = torch.from_numpy(chunk)

            speech_probability = vad_model(
                tensor,
                VAD_RATE,
            ).item()

            is_speech = (
                speech_probability >= VAD_THRESHOLD
            )

            # ------------------------------------------------
            # Speech start
            # ------------------------------------------------

            if not speech_active:

                # Maintain pre-roll
                pre_roll = np.concatenate([
                    pre_roll,
                    chunk,
                ])

                if len(pre_roll) > pre_roll_samples:
                    pre_roll = pre_roll[
                        -pre_roll_samples:
                    ]

                if is_speech:

                    speech_active = True

                    speech_duration_ms = 0
                    silence_duration_ms = 0
                    utterance_duration_ms = 0

                    utterance = [
                        pre_roll.copy()
                    ]

                    logger.info(
                        "[SPEECH START] prob=%.2f",
                        speech_probability,
                    )

            # ------------------------------------------------
            # Speech active
            # ------------------------------------------------

            else:

                utterance.append(chunk)

                utterance_duration_ms += (
                    len(chunk)
                    * 1000
                    / VAD_RATE
                )

                if is_speech:

                    speech_duration_ms += (
                        len(chunk)
                        * 1000
                        / VAD_RATE
                    )

                    silence_duration_ms = 0

                else:

                    silence_duration_ms += (
                        len(chunk)
                        * 1000
                        / VAD_RATE
                    )

                # --------------------------------------------
                # End of utterance
                # --------------------------------------------

                if (
                    silence_duration_ms
                    >= MIN_SILENCE_MS
                ):

                    if (
                        speech_duration_ms
                        >= MIN_SPEECH_MS
                    ):

                        audio_data = np.concatenate(
                            utterance
                        )

                        utterance_number += 1

                        filename = (
                            f"utterance_"
                            f"{utterance_number:04d}.wav"
                        )

                        save_wav(
                            filename,
                            audio_data,
                            VAD_RATE,
                        )

                        logger.info(
                            "[SPEECH END] "
                            "duration=%.0f ms saved=%s",
                            utterance_duration_ms,
                            filename,
                        )

                    else:

                        logger.info(
                            "[IGNORED] speech too short"
                        )

                    speech_active = False

                    utterance = []

                    speech_duration_ms = 0
                    silence_duration_ms = 0
                    utterance_duration_ms = 0

            # ------------------------------------------------
            # Maximum utterance length
            # ------------------------------------------------

            if (
                speech_active
                and utterance_duration_ms
                >= MAX_UTTERANCE_MS
            ):

                audio_data = np.concatenate(
                    utterance
                )

                utterance_number += 1

                filename = (
                    f"utterance_"
                    f"{utterance_number:04d}.wav"
                )

                save_wav(
                    filename,
                    audio_data,
                    VAD_RATE,
                )

                logger.info(
                    "[MAX LENGTH] saved=%s",
                    filename,
                )

                speech_active = False

                utterance = []

                speech_duration_ms = 0
                silence_duration_ms = 0
                utterance_duration_ms = 0


except KeyboardInterrupt:

    logger.info("Stopping...")


finally:

    logger.info("Closing audio stream...")

    stream.stop_stream()
    stream.close()

    audio.terminate()

    logger.info("Audio resources released.")
