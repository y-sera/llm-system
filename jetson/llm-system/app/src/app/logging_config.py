import logging
import os


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
