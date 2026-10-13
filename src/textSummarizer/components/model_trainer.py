
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from textSummarizer.entity.config_entity import ModelTrainerConfig
from textSummarizer.logging import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, params):
        self.config = config
        self.params = params

        # artifacts/model_trainer -> artifacts/data_transformation
        self.dataset_dir = (
            self.config.root_dir.parent
            / "data_transformation"
            / "tokenized_billsum"
        )

    def train(self):
        logger.info("Starting BART model training")

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Tokenized dataset not found: {self.dataset_dir}"
            )

        dataset = load_from_disk(str(self.dataset_dir))

        if "train" not in dataset or "validation" not in dataset:
            raise ValueError(
                "Tokenized dataset must contain train and validation splits."
            )

        # Load BART model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_checkpoint
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_checkpoint
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
        )

        # Read training parameters from params.yaml
        training_params = dict(self.params)

        # Compatibility with older Transformers versions
        if "eval_strategy" in training_params:
            training_params["evaluation_strategy"] = (
                training_params.pop("eval_strategy")
            )

        # Save checkpoints and model under the configured directory
        training_params["output_dir"] = str(
            self.config.trained_model_dir
        )

        # Mixed-precision training requires CUDA for this configuration
        training_params["fp16"] = (
            bool(training_params.get("fp16", False))
            and torch.cuda.is_available()
        )

        training_args = Seq2SeqTrainingArguments(
            **training_params
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
        )

        train_result = trainer.train()

        # Save the final model
        self.config.trained_model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        trainer.save_model(str(self.config.trained_model_dir))

        # Save the tokenizer separately
        self.config.tokenizer_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        tokenizer.save_pretrained(
            str(self.config.tokenizer_dir)
        )

        logger.info("BART model training and saving completed")

        return train_result
