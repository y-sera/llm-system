import os

import numpy as np
import pyaudio

from .audio import (
    debug_log_input_level,
    find_audio_device,
    pcm_to_wav_bytes,
    prepare_vad_block,
    save_wav,
)
from .config import (
    AUDIO_INPUT_RATE,
    CHANNELS,
    FORMAT,
    INPUT_CHUNK,
    OUTPUT_DIR,
    VAD_MODEL_PATH,
)
from .logging_config import logger
from .segmentation import UtteranceSegmenter
from .transcription import (
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    transcribe_wav,
)
from .vad import SileroVAD


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
            AUDIO_INPUT_RATE,
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
        "  sample_rate=%d Hz",
        AUDIO_INPUT_RATE,
    )

    logger.info(
        "  vad_model=%s",
        VAD_MODEL_PATH,
    )

    # --------------------------------------------------------
    # Load the VAD model
    # --------------------------------------------------------

    try:
        vad_model = SileroVAD(
            VAD_MODEL_PATH,
        )

    except Exception:
        logger.exception(
            "Failed to load Silero VAD"
        )

        raise

    # --------------------------------------------------------
    # Initialize PyAudio
    # --------------------------------------------------------

    logger.info("Initializing PyAudio")

    pa = pyaudio.PyAudio()

    logger.info(
        "Audio device count: %d",
        pa.get_device_count(),
    )

    try:
        device_index = find_audio_device(
            pa,
        )

        device_info = pa.get_device_info_by_index(
            device_index,
        )

    except Exception:
        pa.terminate()
        raise

    logger.info(
        "Using input device %d: %s",
        device_index,
        device_info["name"],
    )

    # --------------------------------------------------------
    # Open audio stream
    # --------------------------------------------------------

    try:
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=AUDIO_INPUT_RATE,
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
        "Audio stream opened: rate=%d chunk=%d",
        AUDIO_INPUT_RATE,
        INPUT_CHUNK,
    )

    logger.info("Listening...")

    segmenter = UtteranceSegmenter()
    utterance_id = 1

    try:
        while True:
            try:

                data = stream.read(
                    INPUT_CHUNK,
                    exception_on_overflow=False,
                )

                debug_log_input_level(
                    np.frombuffer(
                        data,
                        dtype=np.int16,
                    )
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

            if len(audio) != INPUT_CHUNK:
                logger.warning(
                    "Unexpected audio chunk size: "
                    "%d samples (expected %d)",
                    len(audio),
                    INPUT_CHUNK,
                )

                continue

            speech_probability = vad_model(
                prepare_vad_block(audio)
            )

            pcm_data = segmenter.update(
                speech_probability,
                data,
            )

            if pcm_data is not None:
                process_recording(
                    pcm_data,
                    utterance_id,
                )

                utterance_id += 1
                vad_model.reset()

    except KeyboardInterrupt:
        logger.info("Stopping...")

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

        logger.info(
            "Audio application stopped"
        )


if __name__ == "__main__":
    main()
