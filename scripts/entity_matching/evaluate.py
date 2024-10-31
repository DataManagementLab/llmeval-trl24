import collections
import logging

import cattrs
import hydra
from omegaconf import DictConfig

from lib.data import get_instances_dir, get_results_dir, get_responses_dir, load_json, dump_json
from lib.evaluation.metrics import ConfusionMatrix, ConfusionMatrixBy
from lib.model.generic import extract_text_from_response

logger = logging.getLogger(__name__)


def get_ground_truth_boolean(response: str) -> bool | None:
    response_parts = response.lower().split()
    if "yes" in response_parts:
        if "no" in response_parts:
            logger.warning(f"'yes' and 'no' in yes/no response '{response}'!")
            return None
        return True
    elif "no" in response_parts:
        return False
    else:
        logger.warning(f"Neither 'yes' nor 'no' in yes/no response '{response}'!")
        return None


@hydra.main(version_base=None, config_path="../../config/entity_matching", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    instances_dir = get_instances_dir(cfg.task_name, cfg.dataset.dataset_name, cfg.exp_name)
    responses_dir = get_responses_dir(cfg.task_name, cfg.dataset.dataset_name, cfg.exp_name)
    results_dir = get_results_dir(cfg.task_name, cfg.dataset.dataset_name, cfg.exp_name, clear=True)

    errors = collections.Counter()
    confusion = ConfusionMatrix.empty()
    confusion_by_match = ConfusionMatrixBy.empty(("match_category", "clean_or_dirty"))
    confusion_by_perturbation = ConfusionMatrixBy.empty(("perturbation_category",))
    for instance_dir in list(sorted(instances_dir.glob("*/"))):
        ground_truth = load_json(instance_dir / "ground_truth.json")
        response = load_json(responses_dir / f"{instance_dir.name}.json")

        text_completion = extract_text_from_response(response)

        if text_completion is None:
            logger.warning(f"evaluation on failed API request ==> skip")
            errors["api_request_failed"] += 1
            continue

        prediction = get_ground_truth_boolean(text_completion)

        if prediction is None:
            logger.warning(f"Parsing yes/no response '{prediction}' failed! ==> Interpret as incorrect.")
            errors["parse_yes_no_failed"] += 1
            prediction = not ground_truth["rows_match"]  # set prediction to opposite of ground truth

        confusion.push(prediction=prediction, ground_truth=ground_truth["rows_match"])

        if cfg.dataset.dataset_name == "pay_to_inv":
            clean_or_dirty = "clean" if ground_truth["perturbation_categories"] == [] else "dirty"
            confusion_by_match.push(
                {
                    "match_category": ground_truth["match_category"],
                    "clean_or_dirty": clean_or_dirty
                },
                prediction,
                ground_truth["rows_match"]
            )
            if ground_truth["perturbation_categories"] == []:
                confusion_by_perturbation.push(
                    {
                        "perturbation_category": "clean"
                    },
                    prediction,
                    ground_truth["rows_match"]
                )
            else:
                for perturbation_category in ground_truth["perturbation_categories"]:
                    confusion_by_perturbation.push(
                        {
                            "perturbation_category": perturbation_category
                        },
                        prediction,
                        ground_truth["rows_match"]
                    )

    logger.info(f"errors: {errors}")

    dump_json(cattrs.unstructure(confusion), results_dir / "confusion.json")

    confusion_by_match_category = {}
    for k, v in confusion_by_match.group_by_key("match_category").items():
        confusion_by_match_category[k] = cattrs.unstructure(v)
    dump_json(confusion_by_match_category, results_dir / "confusion_by_match_category.json")

    clean_confusion_by_match_category = {}
    for k, v in confusion_by_match.group_by_key(
            "match_category",
            filter_key_values={"clean_or_dirty": "clean"}
    ).items():
        clean_confusion_by_match_category[k] = cattrs.unstructure(v)
    dump_json(clean_confusion_by_match_category, results_dir / "clean_confusion_by_match_category.json")

    dirty_confusion_by_match_category = {}
    for k, v in confusion_by_match.group_by_key(
            "match_category",
            filter_key_values={"clean_or_dirty": "dirty"}
    ).items():
        dirty_confusion_by_match_category[k] = cattrs.unstructure(v)
    dump_json(dirty_confusion_by_match_category, results_dir / "dirty_confusion_by_match_category.json")

    confusion_by_perturb_category = {}
    for k, v in confusion_by_perturbation.group_by_key("perturbation_category").items():
        confusion_by_perturb_category[k] = cattrs.unstructure(v)
    dump_json(confusion_by_perturb_category, results_dir / "confusion_by_perturbation_category.json")

    dump_json(dict(errors), results_dir / "errors.json")


if __name__ == "__main__":
    main()
