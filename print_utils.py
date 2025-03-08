from typing import Callable
from logger import logger


rule = "-" * 20


def print_rule(print_handler: Callable = logger.info, rule_length: int = 20):
    print_handler("-" * rule_length)


def print_box(print_handler: Callable = logger.info, rule_length: int = 20):
    """A decorator that prints a box around the output of a function."""

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            print_rule(print_handler=print_handler, rule_length=rule_length)
            func(*args, **kwargs)
            print_rule(print_handler=print_handler, rule_length=rule_length)

        return wrapper

    return decorator
