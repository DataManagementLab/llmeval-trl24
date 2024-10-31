import logging

import hydra
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from lib.data import get_task_dir, dump_str
from lib.plotting.colors import color
from lib.plotting.plot import prepare_plt, save_plt, hatch
from lib.plotting.texts import text

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/entity_matching", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    ####################################################################################################################
    # Table 1: F1 scores at increasing difficulties
    ####################################################################################################################

    table = pd.read_csv(get_task_dir(cfg.task_name) / "increasing_difficulty.csv", index_col="model")
    table.reset_index(inplace=True)
    latex = table.to_latex(column_format="l" + "r" * (len(table.columns) - 1), index=False)
    latex_lines = latex.splitlines()
    latex_lines[2] = " & ".join(r"\textbf{" + part + "}" for part in latex_lines[2][:-3].split(" & ")) + r" \\"
    latex = "\n".join(latex_lines)
    dump_str(latex, get_task_dir(cfg.task_name) / "increasing_difficulty.tex")

    ####################################################################################################################
    # Figure 2: precision and recall for +multi-matches scenario
    ####################################################################################################################

    # draw plot
    prepare_plt(3, 1.5)
    plt.rcParams["font.size"] = 6.5
    plt.subplots_adjust(left=0.1, top=0.9, bottom=0.3, right=0.95)
    res = pd.read_csv(get_task_dir(cfg.task_name) / "precision_recall.csv", index_col="model")
    res.sort_index(key=lambda x: x.map(lambda m: {
        "gpt-3.5-turbo-1106": 0,
        "gpt-4o-mini-2024-07-18": 1,
        "gpt-4o-2024-08-06": 2
    }[m]), inplace=True)

    for offset, (model, row) in zip([-0.25, 0, 0.25], res.iterrows()):
        plt.bar(
            x=[x + offset for x in [1, 2, 3]],
            height=[row["f1_score"], row["precision"], row["recall"]],
            width=0.225,
            color=color(model),
            label=text(model),
            hatch=hatch(model)
        )
        plt.text(x=1 + offset, y=row["f1_score"] + 0.05, s=round_score_text(row["f1_score"]),
                 color=color(model), ha="center", fontsize=5.5)
        plt.text(x=2 + offset, y=row["precision"] + 0.05, s=round_score_text(row["precision"]),
                 color=color(model), ha="center", fontsize=5.5)
        plt.text(x=3 + offset, y=row["recall"] + 0.05, s=round_score_text(row["recall"]),
                 color=color(model), ha="center", fontsize=5.5)

    plt.xlim((0.5, 3.5))
    plt.xticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5], labels=["", "F1", "", "Precision", "", "Recall", ""])
    plt.ylim((0, 1))
    plt.yticks((0, 0.25, 0.5, 0.75, 1), labels=["0", "", "0.5", "", "1.0"])
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(res.index), fontsize=6.5)

    save_plt(get_task_dir(cfg.task_name) / "precision_recall.pdf")

    ####################################################################################################################
    # Figure 3: F1 scores for typical error categories
    ####################################################################################################################

    prepare_plt(3, 1.5)
    plt.rcParams["font.size"] = 6.5
    plt.subplots_adjust(left=0.15, top=0.9, bottom=0.3, right=0.95)
    res = pd.read_csv(get_task_dir(cfg.task_name) / "error_categories.csv", index_col="model")
    model = "gpt-4o-2024-08-06"
    xs = list(range(1, len(res.columns) + 1))
    plt.bar(x=xs, height=res.loc[model].to_list(), width=0.62, color=color(model), hatch=hatch(model))
    for x, value in zip(xs, res.loc[model]):
        plt.text(x=x, y=value + 0.05, s=round_score_text(value), color=color(model), ha="center", fontsize=5.5)

    plt.xlim((xs[0] - 0.5, xs[-1] + 0.5))
    labels_dict = {
        "initial (clean)": "initial\n(clean)",
        "assignment number": "assignment\nnumber",
        "billing number": "billing\nnumber",
        "partner name": "partner\nname",
        "deduction ≤ $0.1": "deduction\n≤ $0.1"
    }
    plt.xticks(xs, labels=[labels_dict[col] for col in res.columns], fontsize=6.5)
    plt.ylim((0, 1))
    plt.yticks((0, 0.25, 0.5, 0.75, 1), labels=["0", "", "0.5", "", "1.0"])
    plt.ylabel("F1 Score")

    save_plt(get_task_dir(cfg.task_name) / "error_categories.pdf")


def round_score(v: float) -> float:
    return round(v, 2)


def round_score_text(v: float) -> str:
    return f"{round_score(v):.2f}"


if __name__ == "__main__":
    main()
