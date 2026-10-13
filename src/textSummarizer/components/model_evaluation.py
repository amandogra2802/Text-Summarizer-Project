
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textSummarizer.entity.config_entity import ModelEvaluationConfig
from textSummarizer.logging import logger
from textSummarizer.utils.common import create_directories


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.dataset_dir = Path(
            "artifacts/data_ingestion/billsum_dataset"
        )

    def evaluate(self, sample_size=100):
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"BillSum dataset not found: {self.dataset_dir}"
            )

        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.config.model_path}"
            )

        logger.info("Loading trained BART model and tokenizer")

        dataset = load_from_disk(str(self.dataset_dir))
        test_dataset = dataset["test"]

        # Use a small sample first; pass None to evaluate the full test set.
        if sample_size is not None:
            count = min(sample_size, len(test_dataset))
            test_dataset = test_dataset.select(range(count))

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.tokenizer_path)
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.config.model_path)
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model.to(device)
        model.eval()

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL", "rougeLsum"],
            use_stemmer=True,
        )

        totals = {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "rougeLsum": 0.0,
        }

        for example in tqdm(test_dataset, desc="Evaluating BART"):
            inputs = tokenizer(
                example["text"],
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=2,
                    do_sample=False,
                )

            prediction = tokenizer.decode(
                generated[0],
                skip_special_tokens=True,
            )

            scores = scorer.score(
                example["summary"],
                prediction,
            )

            for name in totals:
                totals[name] += scores[name].fmeasure

        if len(test_dataset) == 0:
            raise ValueError("The test dataset is empty.")

        results = {
            name: score / len(test_dataset)
            for name, score in totals.items()
        }

        create_directories([
            Path(self.config.metric_file_name).parent
        ])

        with open(self.config.metric_file_name, "w") as file:
            json.dump(results, file, indent=4)

        logger.info(f"ROUGE evaluation results: {results}")
        logger.info(
            f"Evaluation results saved to: "
            f"{self.config.metric_file_name}"
        )

        return results
