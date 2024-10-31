import logging
from typing import Literal

import pandas as pd

from lib.prompting.template import fill_template

logger = logging.getLogger(__name__)


def linearize_table(
        table: pd.DataFrame,
        table_name: str | None,
        *,
        template: str,
        mode: Literal["csv"] | Literal["markdown"],
        csv_params: dict | None = None,
        markdown_params: dict | None = None
) -> str:
    """Linearize the given table.

    Args:
        table: The table to linearize.
        table_name: The name of the table.
        template: The linearization template, which can contain {{table_name}}, {{table}}, and {{newline}}.
        mode: The linearization mode.
        csv_params: The parameters for the pandas to_csv method.
        markdown_params: The parameters for the pandas to_markdown method.

    Returns:
        The linearized table string.
    """
    if mode == "csv":
        lin_table = table.to_csv(**csv_params)
    elif mode == "markdown":
        lin_table = table.to_markdown(**markdown_params)
    else:
        raise AssertionError(f"Unsupported table linearization mode '{mode}'!")

    return fill_template(template, newline="\n", table_name=table_name, table=lin_table)
