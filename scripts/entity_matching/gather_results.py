import logging

import cattrs
import hydra
import pandas as pd
from omegaconf import DictConfig

from lib.data import get_task_dir, load_json
from lib.evaluation.metrics import ConfusionMatrix

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/entity_matching", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # load all results
    all_exp_paths = list(sorted(get_task_dir(cfg.task_name).glob("*/experiments/*/")))
    all_res = pd.DataFrame({"path": all_exp_paths})
    all_res["dataset"] = all_res["path"].apply(lambda p: p.parent.parent.name)
    all_res["model"] = all_res["path"].apply(lambda p: p.name.split("_")[1])
    all_res["schema_mode"] = all_res["path"].apply(lambda p: p.name.split("_")[2])
    all_res["perturbation_mode"] = all_res["path"].apply(lambda p: p.name.split("_")[3])
    all_res["errors"] = all_res["path"].apply(lambda p: load_json(p / "results" / "errors.json"))
    all_res["confusion"] = all_res["path"].apply(
        lambda p: cattrs.structure(load_json(p / "results" / "confusion.json"), ConfusionMatrix)
    )
    all_res["confusion_by_match_category"] = all_res["path"].apply(
        lambda p: {
            k: cattrs.structure(v, ConfusionMatrix) for k, v in
            load_json(p / "results" / "confusion_by_match_category.json").items()
        }
    )
    all_res["clean_confusion_by_match_category"] = all_res["path"].apply(
        lambda p: {
            k: cattrs.structure(v, ConfusionMatrix) for k, v in
            load_json(p / "results" / "clean_confusion_by_match_category.json").items()
        }
    )
    all_res["dirty_confusion_by_match_category"] = all_res["path"].apply(
        lambda p: {
            k: cattrs.structure(v, ConfusionMatrix) for k, v in
            load_json(p / "results" / "dirty_confusion_by_match_category.json").items()
        }
    )
    all_res["confusion_by_perturbation_category"] = all_res["path"].apply(
        lambda p: {
            k: cattrs.structure(v, ConfusionMatrix) for k, v in
            load_json(p / "results" / "confusion_by_perturbation_category.json").items()
        }
    )

    ####################################################################################################################
    # Table 1: F1 scores at increasing difficulties
    ####################################################################################################################

    table = pd.DataFrame(
        index=["gpt-3.5-turbo-1106", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"],
        columns=["initial data", "+ errors", "+ multi-matches", "+ multiple tables"]
    )
    for model in table.index:
        # initial data
        res = all_res.loc[
            (all_res["dataset"] == "pay_to_inv")
            & (all_res["model"] == model)
            & (all_res["schema_mode"] == "opaque")
            & (all_res["perturbation_mode"] == "multi")
            ]
        assert len(res.index) == 1
        res = res.iloc[0]
        confusion = res["clean_confusion_by_match_category"]["one_pay_one_inv"]
        table.at[
            model, "initial data"] = f"{round_score(confusion.f1_score):0.2f} ± {round_score(confusion.bootstrap_f1_score_standard_error()):0.2f}"

        # + errors
        res = all_res.loc[
            (all_res["dataset"] == "pay_to_inv")
            & (all_res["model"] == model)
            & (all_res["schema_mode"] == "opaque")
            & (all_res["perturbation_mode"] == "multi")
            ]
        assert len(res.index) == 1
        res = res.iloc[0]
        confusion = res["dirty_confusion_by_match_category"]["one_pay_one_inv"]
        table.at[
            model, "+ errors"] = f"{round_score(confusion.f1_score):0.2f} ± {round_score(confusion.bootstrap_f1_score_standard_error()):0.2f}"

        # + multi-matches
        res = all_res.loc[
            (all_res["dataset"] == "pay_to_inv")
            & (all_res["model"] == model)
            & (all_res["schema_mode"] == "opaque")
            & (all_res["perturbation_mode"] == "multi")
            ]
        assert len(res.index) == 1
        res = res.iloc[0]
        confusion = res["dirty_confusion_by_match_category"]["one_pay_multi_inv"]
        confusion = confusion + res["dirty_confusion_by_match_category"]["multi_pay_one_inv"]
        table.at[
            model, "+ multi-matches"] = f"{round_score(confusion.f1_score):0.2f} ± {round_score(confusion.bootstrap_f1_score_standard_error()):0.2f}"

        # + multiple tables
        res = all_res.loc[
            (all_res["dataset"] == "pay_to_inv")
            & (all_res["model"] == model)
            & (all_res["schema_mode"] == "multi-table")
            & (all_res["perturbation_mode"] == "multi")
            ]
        assert len(res.index) == 1
        res = res.iloc[0]
        confusion = res["dirty_confusion_by_match_category"]["one_pay_multi_inv"]
        confusion = confusion + res["dirty_confusion_by_match_category"]["multi_pay_one_inv"]
        table.at[
            model, "+ multiple tables"] = f"{round_score(confusion.f1_score):0.2f} ± {round_score(confusion.bootstrap_f1_score_standard_error()):0.2f}"

    table.index.name = "model"
    table.to_csv(get_task_dir(cfg.task_name) / "increasing_difficulty.csv")

    ####################################################################################################################
    # Figure 2: precision and recall for +multi-matches scenario
    ####################################################################################################################

    res = all_res.loc[
        (all_res["dataset"] == "pay_to_inv")
        & (all_res["schema_mode"] == "opaque")
        & (all_res["perturbation_mode"] == "multi")
        ]
    res = res.copy()

    res["f1_score"] = res["dirty_confusion_by_match_category"].apply(
        lambda d: (d["multi_pay_one_inv"] + d["one_pay_multi_inv"]).f1_score
    )
    res["precision"] = res["dirty_confusion_by_match_category"].apply(
        lambda d: (d["multi_pay_one_inv"] + d["one_pay_multi_inv"]).precision
    )
    res["recall"] = res["dirty_confusion_by_match_category"].apply(
        lambda d: (d["multi_pay_one_inv"] + d["one_pay_multi_inv"]).recall
    )

    res = res[["model", "f1_score", "precision", "recall"]]
    res.set_index("model", inplace=True)
    res.to_csv(get_task_dir(cfg.task_name) / "precision_recall.csv")

    ####################################################################################################################
    # Figure 3: F1 scores for typical error categories
    ####################################################################################################################

    res = all_res.loc[
        (all_res["dataset"] == "pay_to_inv")
        & (all_res["schema_mode"] == "opaque")
        & (all_res["perturbation_mode"] == "single")
        ]
    res = res.copy()

    res["initial (clean)"] = res["confusion_by_perturbation_category"].apply(
        lambda d: d["clean"].f1_score
    )
    res["assignment number"] = res["confusion_by_perturbation_category"].apply(
        lambda d: d["perturbed_assignment_number"].f1_score
    )
    res["billing number"] = res["confusion_by_perturbation_category"].apply(
        lambda d: d["perturbed_billing_number"].f1_score
    )
    res["partner name"] = res["confusion_by_perturbation_category"].apply(
        lambda d: d["perturbed_business_partner"].f1_score
    )
    res["deduction ≤ $0.1"] = res["confusion_by_perturbation_category"].apply(
        lambda d: d["small_deduction"].f1_score
    )

    res = res[["model", "initial (clean)", "assignment number", "billing number", "partner name", "deduction ≤ $0.1"]]
    res.set_index("model", inplace=True)
    res.to_csv(get_task_dir(cfg.task_name) / "error_categories.csv")


def round_score(v: float) -> float:
    return round(v, 2)


if __name__ == "__main__":
    main()
