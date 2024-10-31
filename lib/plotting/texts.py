import logging

logger = logging.getLogger(__name__)


def text(name: str) -> str:
    """Return the text assigned to the given name. Default is the name itself.

    Args:
        name: The given name.

    Returns:
        The text.
    """
    if name.startswith("gpt-3.5-turbo"):
        return "GPT-3.5-Turbo"
    elif name.startswith("gpt-4o-mini"):
        return "GPT-4o-Mini"
    elif name.startswith("gpt-4o"):
        return "GPT-4o"
    else:
        return name
