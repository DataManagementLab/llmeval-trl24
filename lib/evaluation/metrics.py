import collections
import logging
import random
import statistics
from typing import Any

import attrs

logger = logging.getLogger(__name__)

bootstrap_random = random.Random(981100968)


@attrs.define
class ConfusionMatrix:
    """ConfusionMatrix matrix with precision, recall, and F1 score."""
    TP: int
    FP: int
    TN: int
    FN: int

    @property
    def total(self) -> int:
        """Total number of instances."""
        return self.TN + self.FP + self.FN + self.TP

    @property
    def precision(self) -> float:
        """Precision score."""
        if self.TP + self.FP == 0:
            return 1
        return self.TP / (self.TP + self.FP)

    @property
    def recall(self) -> float:
        """Recall score."""
        if self.TP + self.FN == 0:
            return 0
        return self.TP / (self.TP + self.FN)

    @property
    def f1_score(self) -> float:
        """F1 score."""
        if self.precision + self.recall == 0:
            return 0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @classmethod
    def empty(cls) -> "ConfusionMatrix":
        """Create an empty confusion matrix object."""
        return cls(0, 0, 0, 0)

    def push(self, prediction: bool, ground_truth: bool) -> None:
        """Include the given instance in the confusion matrix.

        Args:
            prediction: The predicted value.
            ground_truth: The ground truth value.
        """
        self.TP += int(prediction and ground_truth)
        self.FP += int(prediction and not ground_truth)
        self.TN += int(not prediction and not ground_truth)
        self.FN += int(not prediction and ground_truth)

    def __add__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        return ConfusionMatrix(
            self.TP + other.TP,
            self.FP + other.FP,
            self.TN + other.TN,
            self.FN + other.FN
        )

    def bootstrap_f1_score_standard_error(self, *, n_rounds: int = 1_000) -> float:
        """Use bootstrapping to determine the F1 score's standard error.

        Args:
            n_rounds: The number of rounds for bootstrapping.

        Returns:
            The F1 score's standard error.
        """
        f1_scores = []
        population = ["TP", "FP", "TN", "FN"]
        weights = [self.TP / self.total, self.FP / self.total, self.TN / self.total, self.FN / self.total]
        for _ in range(n_rounds):
            counts = {p: 0 for p in population}
            for _ in range(self.total):
                counts[bootstrap_random.choices(population, weights)[0]] += 1
            f1_scores.append(ConfusionMatrix(**counts).f1_score)
        return statistics.stdev(f1_scores)


@attrs.define
class ConfusionMatrixBy:
    """ConfusionMatrix by key."""
    keys: tuple[str, ...]
    mapping: dict[tuple[Any, ...], ConfusionMatrix]

    @classmethod
    def empty(cls, keys: tuple[str, ...]) -> "ConfusionMatrixBy":
        """Create an empty ConfusionMatrixBy object."""
        return cls(keys, collections.defaultdict(ConfusionMatrix.empty))

    def _key_values_dict_to_tuple(self, key_values: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(key_values[key] for key in self.keys)

    def push(self, key_values: dict[str, Any], prediction: bool, ground_truth: bool) -> None:
        """Include the given instance in the confusion for the given key values.

        Args:
            key_values: The key values for the instance.
            prediction: The predicted value.
            ground_truth: The ground truth value.
        """
        self.mapping[self._key_values_dict_to_tuple(key_values)].push(prediction, ground_truth)

    @property
    def all(self) -> ConfusionMatrix:
        """Return the confusion matrix across all key values."""
        res = ConfusionMatrix.empty()
        for confusion in self.mapping.values():
            res += confusion
        return res

    def group_by_key(self, key: str, *, filter_key_values: dict[str, Any] | None = None) -> dict[str, ConfusionMatrix]:
        """Group the accuracy by the given key.

        Args:
            key: The key to group by.
            filter_key_values: An optional dictionary of values to filter by.

        Returns:
            Mapping from key value to accuracy.
        """
        res = collections.defaultdict(ConfusionMatrix.empty)

        for key_values, confusion in self.mapping.items():
            if filter_key_values is not None:
                skip = False
                for k, v in filter_key_values.items():
                    if key_values[self.keys.index(k)] != v:
                        skip = True
                        break
                if skip:
                    continue

            value = key_values[self.keys.index(key)]
            res[value] += confusion

        return res
