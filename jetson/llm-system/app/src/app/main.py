import base64
import io
import logging
import os
import time
import wave

import numpy as np
import onnxruntime as ort
import pyaudio
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

# Microphone now directly captures at the VAD sample rate.
INPUT_RATE = 16000
VAD_RATE = 16000

CHANNELS = 1
FORMAT = pyaudio.paInt16

# 32 ms audio chunks.
# At 16 kHz this is exactly 512 samples, which is the
# Silero VAD input size.
CHUNK_MS = 32
INPUT_CHUNK = int(INPUT_RATE * CHUNK_MS / 1000)

VAD_CHUNK = 512

VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.3"))

# Exit threshold for two-threshold hysteresis. While a recording is
# active the lower VAD_THRESHOLD_END is used so short dips in the
# speech probability do not stop the recording.
VAD_THRESHOLD_END = float(os.getenv("VAD_THRESHOLD_END", "0.2"))

# Linear gain applied ONLY to the float signal handed to the VAD model.
# Does not affect the PCM saved to the WAV file. Only used when the
# automatic VAD normalisation is disabled (VAD_AUTO_NORM=0).
VAD_GAIN = float(os.getenv("VAD_GAIN", "1.0"))

# Saturation guard: fraction of a chunk pinned at the int16 rails above
# which the capture is treated as clipping, so the gain is not applied.
VAD_CLIP_LIMIT = float(os.getenv("VAD_CLIP_LIMIT", "0.02"))

# When enabled each VAD chunk is DC-removed then robustly peak-normalised
# to a common working range, so one build behaves well for both very quiet
# and very hot captures. Disabled falls back to the fixed VAD_GAIN above.
# The saved WAV bytes are unaffected either way.
VAD_AUTO_NORM = os.getenv("VAD_AUTO_NORM", "1") == "1"
VAD_TARGET_PEAK = float(os.getenv("VAD_TARGET_PEAK", "0.6"))
VAD_MAX_NORM_GAIN = float(os.getenv("VAD_MAX_NORM_GAIN", "16.0"))

MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", "150"))
MIN_SILENCE_MS = int(os.getenv("MIN_SILENCE_MS", "700"))

PRE_ROLL_MS = 300

MAX_UTTERANCE_MS = 10000

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    "./recordings",
)

VAD_MODEL_PATH = os.getenv(
    "VAD_MODEL_PATH",
    "/app/models/silero_vad.onnx",
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


# ============================================================
# Silero VAD - ONNX Runtime
# ============================================================

class SileroVAD:
    """
    Silero VAD using ONNX Runtime.

    The standard Silero VAD ONNX model expects:

        input : [1, 512] float32
        state : [2, 1, 128] float32
        sr    : [1] int64

    and returns:

        output : speech probability
        state  : updated recurrent state
    """

    def __init__(
        self,
        model_path: str,
    ):
        logger.info(
            "Loading Silero VAD ONNX model: %s",
            model_path,
        )

        self.session = ort.InferenceSession(
            model_path,
            providers=[
                "CPUExecutionProvider",
            ],
        )

        providers = self.session.get_providers()

        logger.info(
            "Silero VAD ONNX providers: %s",
            providers,
        )

        # Silero VAD recurrent state.
        self.state = np.zeros(
            (2, 1, 128),
            dtype=np.float32,
        )

        self.sample_rate = np.array(
            [VAD_RATE],
            dtype=np.int64,
        )

        # Log model I/O information once.
        for input_meta in self.session.get_inputs():
            logger.debug(
                "VAD input: name=%s shape=%s type=%s",
                input_meta.name,
                input_meta.shape,
                input_meta.type,
            )

        for output_meta in self.session.get_outputs():
            logger.debug(
                "VAD output: name=%s shape=%s type=%s",
                output_meta.name,
                output_meta.shape,
                output_meta.type,
            )

    def reset(self):
        """
        Reset recurrent VAD state.
        """
        self.state.fill(0)

    def __call__(
        self,
        chunk: np.ndarray,
    ) -> float:
        """
        Run VAD inference for one 512-sample chunk.

        Input:
            float32 numpy array
            shape: (512,)
            range: approximately [-1.0, 1.0]

        Returns:
            Speech probability.
        """

        if chunk.dtype != np.float32:
            chunk = chunk.astype(
                np.float32,
                copy=False,
            )

        if chunk.ndim != 1:
            raise ValueError(
                f"Expected 1-D audio chunk, "
                f"got shape={chunk.shape}"
            )

        if len(chunk) != VAD_CHUNK:
            raise ValueError(
                f"Expected {VAD_CHUNK} samples, "
                f"got {len(chunk)}"
            )

        input_data = chunk.reshape(
            1,
            VAD_CHUNK,
        )

        logger.debug(
            "VAD input: shape=%s dtype=%s state_shape=%s",
            input_data.shape,
            input_data.dtype,
            self.state.shape,
        )

        outputs = self.session.run(
            None,
            {
                "input": input_data,
                "state": self.state,
                "sr": self.sample_rate,
            },
        )

        logger.debug(
            "VAD outputs: %s",
            [(o.shape, o.dtype) for o in outputs],
        )

        speech_probability = float(
            np.asarray(outputs[0]).reshape(-1)[0]
        )

        # The second output is the updated recurrent state.
        self.state = outputs[1]

        return speech_probability


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

        # Audio is now captured directly at 16 kHz.
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
        "  input_rate=%d Hz",
        INPUT_RATE,
    )

    logger.info(
        "  vad_rate=%d Hz",
        VAD_RATE,
    )

    logger.info(
        "  vad_model=%s",
        VAD_MODEL_PATH,
    )

    # --------------------------------------------------------
    # Load Silero VAD
    # --------------------------------------------------------

    try:

        vad_model = SileroVAD(
            VAD_MODEL_PATH
        )

    except Exception:

        logger.exception(
            "Failed to load Silero VAD"
        )

        raise

    # --------------------------------------------------------
    # Initialize PyAudio
    # --------------------------------------------------------

    logger.info(
        "Initializing PyAudio"
    )

    pa = pyaudio.PyAudio()

    logger.info(
        "Audio device count: %d",
        pa.get_device_count(),
    )

    try:

        device_index = find_audio_device(pa)

        device_info = pa.get_device_info_by_index(
            device_index
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

    # VAD diagnostic log
    last_vad_log_time = time.monotonic()
    window_max_probability = 0.0

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
                audio_int16 = np.frombuffer(data, dtype=np.int16)

                _f = audio_int16.astype(np.float32)
                _dc = float(_f.mean())
                _ac_rms = float(np.sqrt(np.mean((_f - _dc) ** 2)))
                _clip = float(np.mean(np.abs(_f) >= 32000.0))

                logger.info(
                    "Audio level: min=%d max=%d mean=%.1f "
                    "rms=%.1f ac_rms=%.1f clip=%.3f",
                    int(audio_int16.min()),
                    int(audio_int16.max()),
                    _dc,
                    float(np.sqrt(np.mean(_f ** 2))),
                    _ac_rms,
                    _clip,
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

            # ------------------------------------------------
            # 16 kHz PCM -> float32
            # ------------------------------------------------

            audio = np.frombuffer(
                data,
                dtype=np.int16,
            )

            if len(audio) == 0:

                logger.warning(
                    "Audio input returned zero samples"
                )

                continue

            if len(audio) != VAD_CHUNK:

                logger.warning(
                    "Unexpected audio chunk size: %d samples "
                    "(expected %d)",
                    len(audio),
                    VAD_CHUNK,
                )

                continue

            vad_audio = (
                audio.astype(
                    np.float32
                )
                / 32768.0
            )

            # Remove DC offset (Silero is trained on DC-free speech;
            # a DC bias depresses the speech probability). The saved WAV
            # bytes come from the original PCM and are never touched here.
            vad_audio = vad_audio - vad_audio.mean()

            # Fraction of samples pinned at the int16 rails. Computed on
            # float magnitudes so int16 does not overflow at -32768.
            _f32 = audio.astype(np.float32)
            _clip_ratio = float(
                np.mean(np.abs(_f32) >= 32000.0)
            )

            if VAD_AUTO_NORM:
                # Robust peak target via a high percentile so a handful of
                # clipped samples does not defeat normalisation. The cap
                # keeps a silent noise floor from being boosted without
                # limit. This lifts quiet capture and attenuates hot
                # capture toward one working range for the VAD model.
                _peak = float(
                    np.percentile(np.abs(vad_audio), 99.0)
                )

                if _peak > 1e-4:
                    _scale = min(
                        VAD_TARGET_PEAK / _peak,
                        VAD_MAX_NORM_GAIN,
                    )

                    vad_audio = vad_audio * _scale
            else:
                # Fixed manual gain, but never applied to a clipping chunk
                # (amplifying a square wave only hurts the VAD score).
                if _clip_ratio <= VAD_CLIP_LIMIT:
                    vad_audio = vad_audio * VAD_GAIN

            vad_audio = np.clip(
                vad_audio,
                -1.0,
                1.0,
            )

            # ------------------------------------------------
            # VAD
            # ------------------------------------------------

            is_speech = False
            max_speech_probability = 0.0

            speech_probability = vad_model(
                vad_audio
            )

            max_speech_probability = (
                speech_probability
            )

            # Track the highest probability seen since the last
            # diagnostic log so thresholds can be calibrated.
            if speech_probability > window_max_probability:
                window_max_probability = speech_probability

            # Two-threshold hysteresis: once recording has started,
            # keep using the lower exit threshold so brief dips in the
            # speech probability do not interrupt an utterance.
            if speech_started:
                active_threshold = VAD_THRESHOLD_END
            else:
                active_threshold = VAD_THRESHOLD

            if (
                speech_probability
                >= active_threshold
            ):

                is_speech = True

            # ------------------------------------------------
            # Periodic VAD diagnostic log
            # ------------------------------------------------

            now = time.monotonic()

            if (
                now - last_vad_log_time
                >= 5.0
            ):

                logger.info(
                    "VAD status: speech=%s max_prob=%.3f "
                    "start=%.2f end=%.2f gain=%.1f recording=%s",
                    is_speech,
                    window_max_probability,
                    VAD_THRESHOLD,
                    VAD_THRESHOLD_END,
                    VAD_GAIN,
                    speech_started,
                )

                last_vad_log_time = now
                window_max_probability = 0.0

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
