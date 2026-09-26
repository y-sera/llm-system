import numpy as np
import onnxruntime as ort

from .logging_config import logger
from .config import (
    AUDIO_INPUT_RATE,
    VAD_CHUNK,
)


# ============================================================
# Silero VAD - ONNX Runtime
# ============================================================

class SileroVAD:
    """
    Silero VAD using ONNX Runtime.

    The standard Silero VAD ONNX model expects:

        input : [1, 576] float32  (64-sample context + 512-sample window)
        state : [2, 1, 128] float32
        sr    : scalar int64 (16000)

    and returns:

        output : speech probability
        state  : updated recurrent state

    The trailing 64 samples of each input are carried over and prepended
    as the context of the next call, matching the upstream OnnxWrapper.
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

        # The ONNX model expects every 512-sample window to be preceded by a
        # 64-sample overlap context (4 ms at 16 kHz), exactly like the
        # upstream OnnxWrapper does. Carry that context across chunks.
        self.context_size = 64
        self.context = np.zeros(
            (1, self.context_size),
            dtype=np.float32,
        )

        # "sr" must be a zero-dimensional (scalar) int64 tensor, not [sr].
        self.sample_rate = np.array(
            AUDIO_INPUT_RATE,
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
        Reset recurrent VAD state and the cross-chunk context buffer.
        """
        self.state.fill(0)
        self.context.fill(0)

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

        window = chunk.reshape(
            1,
            VAD_CHUNK,
        )

        # Prepend the previous chunk's trailing context (upstream contract:
        # the model input length is context_size + VAD_CHUNK = 64 + 512).
        input_data = np.concatenate(
            [self.context, window],
            axis=1,
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

        # Feed back the recurrent state and the trailing context window.
        self.state = outputs[1]
        self.context = input_data[:, -self.context_size:].copy()

        return speech_probability
