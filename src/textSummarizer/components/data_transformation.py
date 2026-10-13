from pathlib import Path

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer

from textSummarizer.entity.config_entity import DataTransformationConfig
from textSummarizer.logging import logger
from textSummarizer.utils.common import create_directories


class DataTransformation:
    def __init__(
        self,
        config: DataTransformationConfig,
        model_checkpoint: str,
    ):
        self.config = config
        self.model_checkpoint = model_checkpoint

        self.data_path = Path(
            "artifacts/data_ingestion/billsum_dataset"
        )

        self.tokenizer = None

    def convert_examples_to_features(self, examples):
        # Tokenize the legal documents
        inputs = self.tokenizer(
            examples["text"],
            max_length=self.config.max_input_length,
            truncation=True,
        )

        # Tokenize the reference summaries
        targets = self.tokenizer(
            text_target=examples["summary"],
            max_length=self.config.max_target_length,
            truncation=True,
        )

        inputs["labels"] = targets["input_ids"]

        return inputs

    def transform(self):
        try:
            create_directories([
                self.config.root_dir,
                self.config.tokenized_dataset_dir,
            ])

            if not self.data_path.exists():
                raise FileNotFoundError(
                    f"BillSum dataset not found at {self.data_path}. "
                    "Run data ingestion first."
                )

            # Reuse an existing transformed dataset when available
            output_dir = self.config.tokenized_dataset_dir

            if (output_dir / "dataset_dict.json").exists():
                logger.info("Loading previously tokenized dataset")
                return load_from_disk(str(output_dir))

            # Load the dataset saved by data ingestion
            dataset = load_from_disk(str(self.data_path))

            # Load the tokenizer corresponding to BART
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_checkpoint
            )

            transformed_splits = {}

            for split_name, split_data in dataset.items():
                logger.info(
                    f"Tokenizing {split_name}: "
                    f"{len(split_data)} examples"
                )

                transformed_splits[split_name] = split_data.map(
                    self.convert_examples_to_features,
                    batched=True,
                    remove_columns=split_data.column_names,
                )

            tokenized_dataset = DatasetDict(transformed_splits)

            # Save the transformed dataset
            tokenized_dataset.save_to_disk(str(output_dir))

            logger.info(
                f"Tokenized dataset saved to {output_dir}"
            )

            return tokenized_dataset

        except Exception as error:
            logger.exception(
                f"Data transformation failed: {error}"
            )
            raise