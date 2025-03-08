import logging
from typing import Optional, Literal


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# ANSI color codes for terminal text
RESET = "\033[0m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"


# Custom formatter with colors
class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: BLUE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: f"{RED}\033[1m",  # Bold red for critical
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, RESET)
        log_message = super().format(record)
        return f"{color}{log_message}{RESET}"


def create_console_handler():
    # Console handler with color
    console_handler = logging.StreamHandler()
    console_handler.set_name("console_handler")
    console_handler.setFormatter(
        ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    return console_handler


# Set up the base logger
def create_default_logger(
    name: str,
    has_console_handler,
):
    logger = logging.getLogger(name)

    if has_console_handler:
        console_handler = create_console_handler()
        logger.addHandler(console_handler)

    return logger


def update_logger(
    logger: logging.Logger,
    log_level: Optional[LogLevel] = None,
):
    if log_level is not None:
        logger.setLevel(log_level)


logger = create_default_logger("main-logger", has_console_handler=True)
