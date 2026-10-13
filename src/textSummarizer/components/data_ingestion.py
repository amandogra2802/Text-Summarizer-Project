
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

from textSummarizer.entity.config_entity import DataIngestionConfig
from textSummarizer.logging import logger
from textSummarizer.utils.common import create_directories


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.dataset_dir = self.config.root_dir / "billsum_dataset"

    def download_and_save_dataset(self):
        try:
            create_directories([
                self.config.root_dir,
                self.config.cache_dir,
            ])

            # Reuse the dataset if it was already saved.
            if (self.dataset_dir / "dataset_dict.json").exists():
                logger.info("Loading previously saved BillSum dataset")
                return load_from_disk(str(self.dataset_dir))

            logger.info(
                f"Downloading dataset: {self.config.dataset_name}"
            )

            raw_dataset = load_dataset(
                self.config.dataset_name,
                cache_dir=str(self.config.cache_dir),
            )

            if "train" not in raw_dataset or "test" not in raw_dataset:
                raise ValueError(
                    "BillSum must contain train and test splits."
                )

            # Create a reproducible 90/10 train-validation split.
            split_dataset = raw_dataset["train"].train_test_split(
                test_size=0.1,
                seed=42,
            )

            dataset_splits = {
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
                "test": raw_dataset["test"],
            }

            # Preserve the California test set when available.
            if "ca_test" in raw_dataset:
                dataset_splits["ca_test"] = raw_dataset["ca_test"]

            dataset = DatasetDict(dataset_splits)

            dataset.save_to_disk(str(self.dataset_dir))

            logger.info(
                f"BillSum dataset saved to: {self.dataset_dir}"
            )

            for split_name, split_data in dataset.items():
                logger.info(
                    f"{split_name}: {len(split_data)} examples"
                )

            return dataset

        except Exception as error:
            logger.exception(
                f"Data ingestion failed: {error}"
            )
            raise
