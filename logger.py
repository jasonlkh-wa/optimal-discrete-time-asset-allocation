"""
Logger module for managing and formatting log messages with color.

This module provides functionality to log messages with different levels of severity
(e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL) and to format them with ANSI color codes
for enhanced readability in terminal outputs.

Attributes:
    LogLevel: A type alias for log levels as string literals.
    RESET: ANSI code for resetting color formatting.
    RED: ANSI code for red text.
    YELLOW: ANSI code for yellow text.

Classes:
    ColoredFormatter: Custom logging formatter that supports colored output.

Typical usage example:
    import logging
    from logger import logger, LogLevel

    logger.setLevel(logging.DEBUG)
    logger.debug("This is a debug message")
"""

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
    """Custom logging formatter that supports colored output."""

    LEVEL_COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: BLUE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: f"{RED}\033[1m",  # Bold red for critical
    }

    def format(self, record) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, RESET)
        log_message = super().format(record)
        return f"{color}{log_message}{RESET}"


def create_console_handler() -> logging.StreamHandler:
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
) -> logging.Logger:
    """
    Creates a logger with a specified name and adds a console handler if specified.

    Args:
        name (str): The name of the logger.
        has_console_handler (bool): Whether to add a console handler to the logger.

    Returns:
        logging.Logger: A logger with the specified name and console handler.
    """
    logger = logging.getLogger(name)

    if has_console_handler:
        console_handler = create_console_handler()
        logger.addHandler(console_handler)

    return logger


def update_logger(
    logger: logging.Logger,
    log_level: Optional[LogLevel] = None,
) -> None:
    """
    Updates the log level of the given logger.

    Args:
        logger (logging.Logger): The logger to be updated.
        log_level (Optional[LogLevel], optional): The new log level to set. Defaults to None, which means no change is made.
    """
    if log_level is not None:
        logger.setLevel(log_level)


logger = create_default_logger(
    "main-logger", has_console_handler=True
)  # initialize the logger to be used in other modules
