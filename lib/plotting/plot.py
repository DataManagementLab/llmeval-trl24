import logging
import pathlib

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

FONT_SIZE: int = 8


def prepare_plt(width: float, height: float) -> None:
    """Prepare matplotlib to create a plot.

    Args:
        width: The width of the plot in the paper.
        height: The height of the plot in the paper.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams["figure.figsize"] = (width, height)
    plt.rcParams["font.size"] = FONT_SIZE
    plt.rcParams["hatch.linewidth"] = 0.3


def save_plt(
        path: pathlib.Path
) -> None:
    """Save the plot at the given path.

    Args:
        path: The path at which to save the plot.
    """
    plt.savefig(path)
    plt.clf()


def hatch(name: str) -> str | None:
    """Return the hatch assigned to the given name. Default is None.

    Args:
        name: The given name.

    Returns:
        The hatch.
    """
    if name.startswith("gpt-3.5-turbo"):
        return "++"
    elif name.startswith("gpt-4o-mini"):
        return "///"
    elif name.startswith("gpt-4o"):
        return "\\\\\\"
    else:
        return None
