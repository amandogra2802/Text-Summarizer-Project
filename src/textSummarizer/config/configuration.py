from pathlib import Path

from textSummarizer.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    PredictionConfig,
)

from textSummarizer.utils.common import (
    read_yaml,
    create_directories,
)


class ConfigurationManager:

    def __init__(
        self,
        config_filepath=Path("config/config.yaml"),
        params_filepath=Path("params.yaml"),
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self):
        config = self.config.data_ingestion

        create_directories([config.root_dir, config.cache_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            dataset_name=config.dataset_name,
            cache_dir=Path(config.cache_dir),
        )

    def get_data_validation_config(self):
        config = self.config.data_validation

        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            status_file=Path(config.status_file),
            required_columns=list(config.required_columns),
            required_splits=list(config.required_splits),
        )

    def get_data_transformation_config(self):
        config = self.config.data_transformation

        create_directories([
            config.root_dir,
            config.tokenized_dataset_dir,
        ])

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            tokenized_dataset_dir=Path(
                config.tokenized_dataset_dir
            ),
            max_input_length=config.max_input_length,
            max_target_length=config.max_target_length,
        )

    def get_model_trainer_config(self):
        config = self.config.model_trainer

        create_directories([
            config.root_dir,
            config.trained_model_dir,
            config.tokenizer_dir,
        ])

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            model_checkpoint=config.model_checkpoint,
            trained_model_dir=Path(config.trained_model_dir),
            tokenizer_dir=Path(config.tokenizer_dir),
        )

    def get_model_evaluation_config(self):
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            tokenizer_path=Path(config.tokenizer_path),
            metric_file_name=Path(config.metric_file_name),
        )

    def get_prediction_config(self):
        config = self.config.prediction

        return PredictionConfig(
            model_path=Path(config.model_path),
            tokenizer_path=Path(config.tokenizer_path),
            max_input_length=config.max_input_length,
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
        )