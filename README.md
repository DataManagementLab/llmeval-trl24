# Automating Enterprise Data Engineering with LLMs

**The automation of data engineering tasks is invaluable for enterprises to increase efficiency and reduce the manual
effort associated with handling large amounts of data. Large Language Models (LLMs) have recently shown promising
results in enabling this automation. However, data engineering tasks in real-world enterprise scenarios are often
more complex than their typical formulations in the scientific community. In this paper, we study the challenges
that arise when automating real-world enterprise data engineering tasks with LLMs. As part of the paper, we perform
a case study on the task of matching incoming payments to open invoices, an instance of the entity matching problem.
We also release a hand-crafted dataset based on the actual enterprise scenario to enable the research community to
study the complexity of such enterprise tasks.**

Please check out our [paper](https://openreview.net/pdf?id=m85fYEJtDc)!

## Setup

Make sure you have [Conda](https://docs.conda.io/projects/miniconda/en/latest/) installed.

Create a new Conda environment, activate it, and add the project to the Python path:

```bash
conda env create -f environment.yml
conda activate llmeval-trl24
export PYTHONPATH=${PYTHONPATH}:./
```

## Payment-to-Invoice Matching Dataset

We release our hand-crafted dataset for the entity matching scenario of matching incoming payments to open invoices. We
provide multiple versions of the dataset with different *perturbation modes* and *schema modes*:

Perturbation modes:

* `single`: each payment has zero or one perturbation
* `multi`: each payment can have multiple perturbations

Schema modes:

* `descriptive`: the columns have descriptive names (e.g., *Client*, *Billing Number*, etc.)
* `opaque`: the columns have opaque names (e.g., *MANDT*, *VBELN*, etc.)
* `multi-table`: the columns have opaque names and invoices are represented by rows in multiple tables

The dataset is located in `data/entity_matching/pay_to_inv/download/<perturbation_mode>/<schema_mode>`. Invoices are
stored in `invoices.csv` and payments in `payments.csv`. Which payments belong to which invoices is stored in
`matches.csv`, which also contains the *match categories* (1:1, 1:N, N:1) and *perturbation categories* (e.g.,
*small_deduction*).

The dataset generator is implemented in `scripts/entity_matching/pay_to_inv/download.py` and configurable in
`config/entity_matching/dataset/pay_to_inv.yaml`.

To re-create the dataset, run:

```bash
bash scripts/entity_matching/create_dataset.sh
```

## Reproducibility

We provide code to create the dataset and reproduce the results from the paper.

Reproducing the exact results from the paper requires the following artifacts:

* `openai_cache.zip` the OpenAI API requests and responses, which you must unpack into `data/openai_cache`

To create the dataset and reproduce the results from the paper, run:

```bash
bash reproduce.sh
```

The results are:

* `data/entity_matching/increasing_difficulty.csv` Table 1 (F1 scores at increasing difficulties)
* `data/entity_matching/precision_recall.pdf` Figure 2 (precision and recall for *+ multi-matches* scenario)
* `data/entity_matching/error_categories.pdf` Figure 3 (F1 scores for typical error categories)