## NTCIR-19 ModelRetrieval Task - Sub Task B

Resources and baseline materials for **Sub Task B** of the **NTCIR-19 ModelRetrieval Task**.

- Official website: https://modelretrieval-1.github.io/

## Overview

Given a query image, your system must rank image style transfer LoRA models by their ability to reproduce the style of that image.

In other words, this is a model retrieval problem where each query corresponds to one correct generating model.

## Dataset Structure

```text
data/
  train/
    {model_id}/
      train-{content_image_id}-{model_id}.png
  test/
    test-{content_image_id}-{identifier}.png

  models.csv
  content-images.csv
  train.csv
  test.csv

notebooks/
  baseline.ipynb
```

## Data Description

| File | Description |
|---|---|
| `data/models.csv` | List of image style transfer LoRA models. |
| `data/content-images.csv` | List of content images used to generate query images. |
| `data/train.csv` | Training metadata with `image_id`, `model_id`, `content_image_id`, and `best_seed`. |
| `data/test.csv` | Test metadata for evaluation queries. |

### Train CSV Columns

- `image_id`: generated image ID
- `model_id`: ground-truth model ID
- `content_image_id`: source content image ID
- `best_seed`: seed used to generate the image

## Baseline

- `notebooks/baseline.ipynb`
  - Baseline approach using image classification.

## Quick Start

### 1. Install uv

Follow the official guide:
https://docs.astral.sh/uv/getting-started/installation/

### 2. Sync dependencies

```bash
uv sync
```

### 3. Set up Unsplash API and download content images

Before downloading content images, prepare your Unsplash API credentials:

1. Create an Unsplash account: https://unsplash.com/join
2. Create an Unsplash app: https://unsplash.com/oauth/applications
3. Copy your **Access Key** from the app settings

Then run the downloader command.

Download specific content images:

```bash
python main.py download_content_images --image-id 1 --image-id 2 --access-key <YOUR_UNSPLASH_ACCESS_KEY>
```

Download all content images:

```bash
python main.py download_content_images --all --access-key <YOUR_UNSPLASH_ACCESS_KEY>
```

You can also set the key as an environment variable:

```bash
export UNSPLASH_ACCESS_KEY=<YOUR_UNSPLASH_ACCESS_KEY>
python main.py download_content_images --all
```

Downloaded files are saved to `data/content-images/` using zero-padded IDs such as `01.jpg`, `02.jpg`, `03.jpg`, etc.

### 4. Set up Civitai token and download models

Before downloading models, prepare your Civitai credentials:

1. Create a Civitai account: https://civitai.com/
2. Generate a Civitai API token from your account settings

Then run the model downloader command.

Download specific models:

```bash
python main.py download_models --civitai-token <YOUR_CIVITAI_TOKEN> --model-id 1 --model-id 02 --model-id 10
```

Download all models:

```bash
python main.py download_models --civitai-token <YOUR_CIVITAI_TOKEN>
```

Downloaded model files are saved to `data/models/` using zero-padded IDs such as `01.safetensors`, `02.safetensors`, `03.safetensors`, etc.

### 5. Run the baseline notebook

Open and run:

- `notebooks/baseline.ipynb`

## Submission Format

Submissions must follow **TREC_EVAL** format:

```text
topicID Q0 docID Rank Score RunID
```

Field definitions:

- `topicID`: query image ID
- `Q0`: fixed token (`Q0`)
- `docID`: model ID
- `Rank`: rank position (1 = highest)
- `Score`: predicted relevance score
- `RunID`: your run identifier

Example:

```text
1 Q0 1 1 0.99 Run01
1 Q0 7 2 0.95 Run01
...
```

## Evaluation

This subtask is evaluated using **MRR (Mean Reciprocal Rank)**.

For each query, take the rank of the correct model and compute reciprocal rank. MRR is the mean over all queries:

$$
\mathrm{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\mathrm{rank}_q}
$$

Where:

- $|Q|$ is the number of queries
- $\mathrm{rank}_q$ is the rank position of the correct model for query $q$

Higher is better. The maximum possible score is $1.0$ when every correct model is ranked first.


