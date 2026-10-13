from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    cache_dir: Path


@dataclass
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    required_columns: List[str]
    required_splits: List[str]


@dataclass
class DataTransformationConfig:
    root_dir: Path
    tokenized_dataset_dir: Path
    max_input_length: int
    max_target_length: int


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    model_checkpoint: str
    trained_model_dir: Path
    tokenizer_dir: Path


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path


@dataclass
class PredictionConfig:
    model_path: Path
    tokenizer_path: Path
    max_input_length: int
    max_new_tokens: int
    num_beams: int