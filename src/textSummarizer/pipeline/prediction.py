
from pathlib import Path

import torch
from transformers import BartTokenizer, AutoModelForSeq2SeqLM

from textSummarizer.entity.config_entity import PredictionConfig
from textSummarizer.logging import logger


class PredictionPipeline:
    def __init__(self, config: PredictionConfig):
        self.config = config

        model_path = Path(config.model_path)
        tokenizer_path = Path(config.tokenizer_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}"
            )

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}"
            )

        # Load the fine-tuned BART model and tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path)
        )

        # Select the available device
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

        logger.info(
            f"Prediction pipeline loaded on {self.device}"
        )

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

        summary = self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        )

        return summary

    def __call__(self, legal_text: str) -> str:
        return self.predict(legal_text)
