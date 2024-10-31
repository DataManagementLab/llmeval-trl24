import logging

import tiktoken

from lib.model._openai import openai_execute

logger = logging.getLogger(__name__)

FORCE: float = 0.05


def num_tokens(
        text: str,
        model: str,
        api_name: str
) -> int:
    """Compute the number of tokens of the text.

    Args:
        text: The given text.
        model: The name of the model.
        api_name: The name of the API to use.

    Returns:
        The number of tokens.
    """
    if api_name == "openai":
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    else:
        raise AssertionError(f"Unknown API name '{api_name}'!")


def execute_requests(
        requests: list[dict],
        api_name: str
) -> list[dict]:
    """Execute the list of requests against the specified API.

    Args:
        requests: The list of API requests.
        api_name: The name of the API.

    Returns:
        The list of API responses.
    """
    if api_name == "openai":
        return openai_execute(requests, force=FORCE)
    else:
        raise AssertionError(f"Unknown API name '{api_name}'!")


def extract_text_from_response(response: dict) -> str | None:
    """Extract the text from an API response.

    Args:
        response: The API response.

    Returns:
        The generated text or None if the API request failed.
    """
    if "choices" not in response.keys():
        return None

    return response["choices"][0]["message"]["content"]


def max_tokens_for_ground_truth(ground_truth: str, api_name: str, model: str,
                                max_tokens_over_ground_truth: int | None) -> int | None:
    """Compute max_tokens based on the length of the ground truth and max_tokens_over_ground_truth.

    Args:
        ground_truth: The ground truth string.
        api_name: The name of the API.
        model: The model name.
        max_tokens_over_ground_truth: How many additional tokens should be allowed.

    Returns:
        The value for max_tokens.
    """
    ground_truth_len = num_tokens(ground_truth, model, api_name)
    return None if max_tokens_over_ground_truth is None else (ground_truth_len + max_tokens_over_ground_truth)
