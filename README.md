## NTCIR-19 ModelRetrieval Task: Sub Task B

This repository contains resources for **Sub Task B** of the **NTCIR-19 ModelRetrieval Task**.

- Official website: https://modelretrieval-1.github.io/

## Task Description

Given a **query image**, the goal is to rank **image style transfer LoRA models** by how well they can reproduce the style of that image.

## Repository Structure

```text
data/
	train/
		{model_id}/
			{image_id}.png
	test/
		{image_id}.png

	models.csv
	content-images.csv
	train.csv
	test.csv

notebooks/
	baseline.ipynb
```

### Data Files

- `data/models.csv`
	- List of image style transfer LoRA models.

- `data/content-images.csv`
	- List of content images used to generate query images.

- `data/train.csv`
	- Training metadata:
		- `image_id`: generated image ID
		- `model_id`: ground-truth label
		- `content_image_id`: source content image ID
		- `best_seed`: seed used to generate the image

- `data/test.csv`
	- Test metadata for evaluation queries.

### Baseline Notebook

- `notebooks/baseline.ipynb`
	- Baseline approach using image classification.

## Getting Started

1. Install `uv`:
	 - https://docs.astral.sh/uv/getting-started/installation/

2. Sync dependencies:

	 ```bash
	 uv sync
	 ```

## Submission Format

Use **TREC_EVAL** format:

```text
topicID Q0 docID Rank Score RunID
```

- `topicID`: query image ID
- `Q0`: fixed string (`Q0`)
- `docID`: model ID
- `Rank`: ranking position of the model
- `Score`: predicted score for the model
- `RunID`: your run identifier

### Example

```text
1 Q0 1 1 0.99 Run01
2 Q0 7 2 0.95 Run01
...
```

## Evaluation

This subtask uses **MRR (Mean Reciprocal Rank)** for evaluation.

For each query image, find the rank position of the correct model and compute its reciprocal rank.

$$
\mathrm{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\mathrm{rank}_q}
$$

- $|Q|$: number of query images
- $\mathrm{rank}_q$: rank position of the correct model for query $q$


