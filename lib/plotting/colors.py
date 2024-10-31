COLOR_BLACK = "#000000"
COLOR_GREY = "#777777"
COLOR_WHITE = "#FFFFFF"

COLOR_1A = "#5D85C3"
COLOR_2A = "#009CDA"
COLOR_3A = "#50B695"
COLOR_4A = "#AFCC50"
COLOR_5A = "#DDDF48"
COLOR_6A = "#FFE05C"
COLOR_7A = "#F8BA3C"
COLOR_8A = "#EE7A34"
COLOR_9A = "#E9503E"
COLOR_10A = "#C9308E"
COLOR_11A = "#804597"


def color(name: str) -> str:
    """Return the color assigned to the given name. Default is black.

    Args:
        name: The given name.

    Returns:
        The color string.
    """
    if name.startswith("gpt-3.5-turbo"):
        return COLOR_1A
    elif name.startswith("gpt-4o-mini"):
        return COLOR_7A
    elif name.startswith("gpt-4o"):
        return COLOR_9A
    else:
        return COLOR_BLACK
