import os
from pathlib import Path

import boto3
import torch
from transformers import BartTokenizer, AutoModelForSeq2SeqLM

from textSummarizer.entity.config_entity import PredictionConfig
from textSummarizer.logging import logger


def required_files_exist(model_path: Path, tokenizer_path: Path) -> bool:
    """Check whether the model and tokenizer are available locally."""
    return (
        (model_path / "config.json").is_file()
        and (model_path / "model.safetensors").is_file()
        and (tokenizer_path / "vocab.json").is_file()
        and (tokenizer_path / "merges.txt").is_file()
    )


def download_model_from_s3() -> tuple[Path, Path]:
    """Download the private BART model and tokenizer from S3."""
    bucket = os.getenv("MODEL_S3_BUCKET")
    prefix = os.getenv("MODEL_S3_PREFIX", "bart-billsum").strip("/")

    if not bucket:
        raise RuntimeError(
            "Model files are missing locally. "
            "Set the MODEL_S3_BUCKET environment variable."
        )

    region = (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "ap-southeast-2"
    )

    cache_dir = Path(
        os.getenv("MODEL_CACHE_DIR", "/tmp/legal-text-summarizer")
    )
    model_path = cache_dir / "model"
    tokenizer_path = cache_dir / "tokenizer"

    model_path.mkdir(parents=True, exist_ok=True)
    tokenizer_path.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=region)

    model_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
    ]

    tokenizer_files = [
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
    ]

    logger.info("Downloading BART model from private S3 storage.")

    for filename in model_files:
        s3.download_file(
            bucket,
            f"{prefix}/{filename}",
            str(model_path / filename),
        )

    for filename in tokenizer_files:
        s3.download_file(
            bucket,
            f"{prefix}/tokenizer/{filename}",
            str(tokenizer_path / filename),
        )

    logger.info("BART model and tokenizer downloaded successfully.")

    return model_path, tokenizer_path


class PredictionPipeline:
    def __init__(self, config: PredictionConfig):
        self.config = config

        model_path = Path(config.model_path)
        tokenizer_path = Path(config.tokenizer_path)

        # Prefer local files; otherwise download from private S3.
        if not required_files_exist(model_path, tokenizer_path):
            model_path, tokenizer_path = download_model_from_s3()

        logger.info("Loading BART tokenizer.")
        self.tokenizer = BartTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
        )

        logger.info("Loading fine-tuned BART model.")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            local_files_only=True,
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Prediction pipeline loaded on {self.device}")

    def predict(self, legal_text: str) -> str:
        if not isinstance(legal_text, str) or not legal_text.strip():
            raise ValueError(
                "Please provide a non-empty legal document."
            )

        inputs = self.tokenizer(
            legal_text,
            max_length=self.config.max_input_length,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_tokens = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
                do_sample=False,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        )

    def __call__(self, legal_text: str) -> str:
        return self.predict(legal_text)