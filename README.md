# Legal Text Summarization using BART and BillSum

An end-to-end NLP project that generates abstractive summaries of legal documents. The project uses the **BART** sequence-to-sequence model and the **BillSum** dataset, with a modular Python codebase and a local FastAPI web interface.

> **Important:** Legal summaries are generated automatically and may omit or misstate critical details. Always verify generated summaries against the original legal text; this application is not legal advice.

## Project Overview

Legal bills can be long and contain formal language, cross-references, and detailed provisions. This project explores abstractive summarization to produce shorter versions of legal documents while retaining their main points.

### Features

- Uses the Hugging Face `facebook/bart-large-cnn` checkpoint as the starting model.
- Uses the BillSum dataset (`FiscalNote/billsum`) for legal-document summarization.
- Provides configurable data ingestion, validation, tokenization, training, and evaluation components.
- Evaluates generated summaries with ROUGE metrics.
- Exposes a FastAPI web page where a user can paste a legal document and request a summary.
- Keeps large model weights and generated artifacts out of ordinary Git commits.

## Technology Stack

- Python 3.10
- PyTorch
- Hugging Face Transformers and Datasets
- BillSum
- ROUGE (`evaluate` / `rouge_score`)
- FastAPI and Uvicorn
- Conda, YAML configuration, Git, and GitHub

## Project Workflow

The codebase is organized around these stages:

1. **Configuration:** `config/config.yaml` and `params.yaml`
2. **Entity definitions:** dataclasses for component configuration
3. **Configuration manager:** loads YAML values and constructs component configuration objects
4. **Data ingestion:** downloads BillSum and creates a validation split from the training split
5. **Data validation:** checks expected columns and dataset splits
6. **Data transformation:** tokenizes legal text and reference summaries for BART
7. **Model training:** fine-tunes a sequence-to-sequence model
8. **Model evaluation:** calculates ROUGE scores
9. **Prediction pipeline:** generates a summary for a supplied legal document
10. **Web application:** serves the summarizer through FastAPI

## Dataset

The project uses [BillSum](https://huggingface.co/datasets/FiscalNote/billsum), a dataset of U.S. legislation with reference summaries.

The workflow creates a validation set by holding out 10% of the original training split. The dataset split sizes observed in the project were:

| Split | Examples | Use |
|---|---:|---|
| Train | 17,054 | Fine-tuning |
| Validation | 1,895 | During-training evaluation |
| Test | 3,269 | Main held-out evaluation |
| California test (`ca_test`) | 1,237 | Additional held-out evaluation |

The exact counts can vary if the upstream dataset changes.

## Model and Preprocessing

- **Base checkpoint:** `facebook/bart-large-cnn`
- **Input column:** `text`
- **Reference-summary column:** `summary`
- **Maximum input length:** 1,024 tokens
- **Maximum target length during tokenization:** 128 tokens
- **Prediction generation:** beam search, configured through the prediction component

Long inputs are truncated to the configured maximum length. Therefore, information near the end of a long legal document may not be included in the model input.

## Evaluation

A Colab experiment evaluated the model on the full BillSum `test` split of 3,269 documents and reported these ROUGE scores:

| Metric | Score |
|---|---:|
| ROUGE-1 | 0.4837 |
| ROUGE-2 | 0.2996 |
| ROUGE-L | 0.3674 |
| ROUGE-Lsum | 0.4042 |

These are ROUGE overlap scores, **not accuracy percentages**. ROUGE measures similarity to reference summaries and does not establish factual or legal correctness.

**Checkpoint note:** The final in-memory training state was not found after a Colab runtime reset. A recoverable `checkpoint-1000` was restored for the local application. The results above are the values recorded by the earlier Colab evaluation and should not be assumed to describe that recovered checkpoint exactly unless the checkpoint is re-evaluated.

## Repository Structure

```text
Text-Summarizer-Project/
├── config/
│   └── config.yaml
├── research/
│   ├── Text-Summarization.ipynb
│   └── trials.ipynb
├── src/
│   └── textSummarizer/
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       ├── config/
│       │   └── configuration.py
│       ├── entity/
│       │   └── config_entity.py
│       ├── pipeline/
│       │   └── prediction.py
│       ├── utils/
│       │   └── common.py
│       └── logging/
├── artifacts/                 # Generated locally; excluded from Git
├── app.py                     # FastAPI application
├── main.py
├── params.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## Setup and Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/amandogra2802/Text-Summarizer-Project.git
cd Text-Summarizer-Project
```

### 2. Create and activate the Conda environment

```bash
conda create -n textS python=3.10 -y
conda activate textS
```

### 3. Install dependencies and the project package

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 4. Place the model and tokenizer files

Large model files and generated artifacts are deliberately excluded from GitHub. Before using the prediction endpoint, place a compatible fine-tuned BART model and tokenizer in the directories configured in `config/config.yaml`. The default paths used in this project are:

```text
artifacts/model_trainer/bart-billsum/  # model config and model weights
artifacts/model_trainer/tokenizer/     # tokenizer files
```

The model folder should include the model weights (for example, `model.safetensors`) and configuration files. The tokenizer folder should include the tokenizer configuration and vocabulary files. If these assets are unavailable, the web page may start but summary generation will fail.

### 5. Start the FastAPI application

From the repository root, run:

```bash
uvicorn app:app --reload
```

Open the web interface at <http://127.0.0.1:8000>.

Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

The `/summarize` endpoint accepts JSON in this form:

```json
{
  "legal_text": "Paste the legal document text here."
}
```

## Research Notebook

The `research/` folder contains the Colab experiment notebook used to explore BillSum, tokenize the data, fine-tune BART, generate example summaries, and evaluate ROUGE. The notebook requires a suitable runtime and dependencies. For training, a GPU runtime is recommended.

## Model Artifacts and GitHub

Model weights, optimizer checkpoints, and generated datasets are not stored in this Git repository because they can be very large. Obtain or train a compatible model separately, then place it in the paths specified above. Do not commit credentials, private access tokens, or large checkpoint files to GitHub.

## Deployment Status

The local FastAPI application has been tested. **AWS deployment and GitHub Actions CI/CD are not configured yet.** The planned deployment workflow is:

1. Create an AWS identity with limited deployment permissions.
2. Create an Amazon ECR repository for the Docker image.
3. Prepare an EC2 host or another suitable compute service.
4. Build and push the image with GitHub Actions.
5. Configure the running application to retrieve the model artifacts securely.
6. Add monitoring and cost alerts, and remove resources when they are no longer needed.

Cloud resources may incur charges. Review AWS pricing and configure billing alerts before creating resources.

## Limitations and Future Improvements

- Improve factual consistency and evaluate important legal details with human review.
- Compare BART with other summarization models and document the experimental settings.
- Re-evaluate the exact checkpoint used by the application and record reproducible results.
- Improve handling of long legal documents, which may exceed the input token limit.
- Add automated tests and CI/CD deployment after model and artifact storage are configured.

## Author

**Aman Dogra**

GitHub: [amandogra2802](https://github.com/amandogra2802)

Repository: [Text-Summarizer-Project](https://github.com/amandogra2802/Text-Summarizer-Project)
