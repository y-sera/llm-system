import io
import os
import wave

import numpy as np

from .logging_config import logger
from .config import (
    VAD_AUTO_NORM,
    VAD_TARGET_PEAK,
    VAD_MAX_NORM_GAIN,
    VAD_GAIN,
    VAD_CLIP_LIMIT,
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


def prepare_vad_block(
    audio: np.ndarray,
) -> np.ndarray:

    vad_audio = audio.astype(np.float32) / 32768.0

    if VAD_AUTO_NORM:
        vad_audio = vad_audio - vad_audio.mean()

        _peak = float(
            np.percentile(np.abs(vad_audio), 99.0)
        )

        if _peak > 1e-4:
            _scale = min(
                VAD_TARGET_PEAK / _peak,
                VAD_MAX_NORM_GAIN,
            )

            vad_audio = vad_audio * _scale

    elif VAD_GAIN != 1.0:
        _clip_ratio = float(
            np.mean(
                np.abs(vad_audio)
                >= (32000.0 / 32768.0)
            )
        )

        if _clip_ratio <= VAD_CLIP_LIMIT:
            vad_audio = vad_audio * VAD_GAIN

    vad_audio = np.clip(
        vad_audio,
        -1.0,
        1.0,
    )

    return vad_audio


def debug_log_input_level(
    audio_int16: np.ndarray,
) -> None:

    _f = audio_int16.astype(np.float32)
    _dc = float(_f.mean())
    _ac_rms = float(np.sqrt(np.mean((_f - _dc) ** 2)))
    _clip = float(np.mean(np.abs(_f) >= 32000.0))

    logger.debug(
        "Audio level: min=%d max=%d mean=%.1f "
        "rms=%.1f ac_rms=%.1f clip=%.3f",
        int(audio_int16.min()),
        int(audio_int16.max()),
        _dc,
        float(np.sqrt(np.mean(_f ** 2))),
        _ac_rms,
        _clip,
    )
