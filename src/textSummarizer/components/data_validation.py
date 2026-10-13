from pathlib import Path

from datasets import load_from_disk

from textSummarizer.entity.config_entity import DataValidationConfig
from textSummarizer.logging import logger
from textSummarizer.utils.common import create_directories


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

        # Location where DataIngestion saved BillSum
        self.dataset_dir = Path(
            "artifacts/data_ingestion/billsum_dataset"
        )

    def validate_all_files(self) -> bool:
        try:
            create_directories([
                self.config.root_dir,
                self.config.status_file.parent,
            ])

            # Load the dataset saved by the ingestion component
            if not self.dataset_dir.exists():
                raise FileNotFoundError(
                    f"Dataset directory not found: {self.dataset_dir}"
                )

            dataset = load_from_disk(str(self.dataset_dir))

            # Check that all required splits exist
            available_splits = set(dataset.keys())
            required_splits = set(self.config.required_splits)

            missing_splits = required_splits - available_splits

            if missing_splits:
                raise ValueError(
                    f"Missing dataset splits: {sorted(missing_splits)}"
                )

            # Check columns and ensure splits are not empty
            required_columns = set(self.config.required_columns)

            for split_name, split_data in dataset.items():
                if len(split_data) == 0:
                    raise ValueError(
                        f"Dataset split '{split_name}' is empty."
                    )

                actual_columns = set(split_data.column_names)
                missing_columns = required_columns - actual_columns

                if missing_columns:
                    raise ValueError(
                        f"Split '{split_name}' is missing columns: "
                        f"{sorted(missing_columns)}"
                    )

                logger.info(
                    f"{split_name}: {len(split_data)} examples validated"
                )

            # Record successful validation
            with open(self.config.status_file, "w") as file:
                file.write("Validation status: True\n")

            logger.info("BillSum dataset validation passed.")
            return True

        except Exception as error:
            create_directories([self.config.status_file.parent])

            with open(self.config.status_file, "w") as file:
                file.write("Validation status: False\n")
                file.write(f"Error: {error}\n")

            logger.exception(f"Dataset validation failed: {error}")
            raise