import time

from .logging_config import logger
from .config import (
    CHUNK_MS,
    MAX_UTTERANCE_MS,
    MIN_SILENCE_MS,
    MIN_SPEECH_MS,
    PRE_ROLL_MS,
    VAD_GAIN,
    VAD_THRESHOLD,
    VAD_THRESHOLD_END,
)


class UtteranceSegmenter:
    """
    Split the incoming chunk stream into discrete utterances.

    update() is fed one (speech_probability, raw_chunk) pair at a time and
    owns the two-threshold hysteresis, the pre-roll buffer, the trailing
    silence timeout and the maximum-utterance cutoff. It returns the joined
    int16 PCM of an utterance on the chunk that completes it, or None while
    a recording is still in progress. The caller owns the VAD model and is
    responsible for resetting it whenever a non-None utterance is returned.
    """

    def __init__(self):
        self.min_speech_frames = int(
            MIN_SPEECH_MS / CHUNK_MS
        )
        self.min_silence_frames = int(
            MIN_SILENCE_MS / CHUNK_MS
        )
        self.max_utterance_frames = int(
            MAX_UTTERANCE_MS / CHUNK_MS
        )
        self.pre_roll_frame_count = int(
            PRE_ROLL_MS / CHUNK_MS
        )

        self.vad_threshold = VAD_THRESHOLD
        self.vad_threshold_end = VAD_THRESHOLD_END
        self.vad_gain = VAD_GAIN

        self.speech_started = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.utterance_frames = []
        self.pre_roll_frames = []

        self.last_vad_log_time = time.monotonic()
        self.window_max_probability = 0.0

    def update(
        self,
        speech_probability: float,
        data: bytes,
    ) -> bytes | None:
        completed = None

        if speech_probability > self.window_max_probability:
            self.window_max_probability = speech_probability

        if self.speech_started:
            active_threshold = self.vad_threshold_end
        else:
            active_threshold = self.vad_threshold

        is_speech = speech_probability >= active_threshold

        now = time.monotonic()

        if now - self.last_vad_log_time >= 5.0:
            logger.info(
                "VAD status: speech=%s max_prob=%.3f "
                "start=%.2f end=%.2f gain=%.1f recording=%s",
                is_speech,
                self.window_max_probability,
                self.vad_threshold,
                self.vad_threshold_end,
                self.vad_gain,
                self.speech_started,
            )

            self.last_vad_log_time = now
            self.window_max_probability = 0.0

        self.pre_roll_frames.append(data)

        if len(self.pre_roll_frames) > self.pre_roll_frame_count:
            self.pre_roll_frames.pop(0)

        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0

            if not self.speech_started:
                if self.speech_frames >= self.min_speech_frames:
                    self.speech_started = True
                    self.utterance_frames = (
                        self.pre_roll_frames.copy()
                    )

                    logger.info("Recording started")

            else:
                self.utterance_frames.append(data)

        else:
            self.silence_frames += 1

            if self.speech_started:
                self.utterance_frames.append(data)

                if (
                    self.silence_frames
                    >= self.min_silence_frames
                ):
                    logger.info("Recording ended")

                    completed = b"".join(
                        self.utterance_frames
                    )

                    self._reset()

        if (
            self.speech_started
            and len(self.utterance_frames)
            >= self.max_utterance_frames
        ):
            logger.warning(
                "Maximum utterance length reached"
            )

            logger.info("Recording ended")

            completed = b"".join(
                self.utterance_frames
            )

            self._reset()

        return completed

    def _reset(self):
        self.speech_started = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.utterance_frames = []