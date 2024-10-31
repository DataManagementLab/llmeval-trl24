import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def unique(generator: Callable, prev_values: list[Any]) -> Any:
    """Calls the generator function until a new value is generated.

    Raises if no new value is generated in 10000 tries.

    Args:
        generator: The generator function, which receives no arguments.
        prev_values: The list of previous values.

    Returns:
        The new value.
    """
    prev_values = set(prev_values)
    for _ in range(10000):
        value = generator()
        if value not in prev_values:
            return value
    raise AssertionError("Unable to generate a unique value in 10000 tries.")
