"""
MLflow Experiment Tracking & Model Registry Integration.

Provides logging for:
- Hyperparameters & training configuration
- Per-epoch training & validation metrics (Loss, Dice, IoU, Precision, Recall)
- Checkpoint artifacts, evaluation reports, and visual overlay samples
- PyTorch Model serialization with MLflow model registry support
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import mlflow
import mlflow.pytorch


class MLflowTracker:
    """
    Manages MLflow experiment lifecycle for Solar Filament ML experiments.
    """

    def __init__(
        self,
        experiment_name: str = "Solar-Filament-Segmentation",
        tracking_uri: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            db_path = Path("mlflow.db").resolve()
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        mlflow.set_experiment(self.experiment_name)
        self.active_run = None

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Starts a new MLflow tracking run."""
        default_tags = {
            "project": "Solar-Filament",
            "framework": "PyTorch",
            "dataset": "MAGFiLO_1.0"
        }
        if tags:
            default_tags.update(tags)

        self.active_run = mlflow.start_run(run_name=run_name, tags=default_tags)
        print(f"[MLflow] Started Run: {self.active_run.info.run_name} (ID: {self.active_run.info.run_id})")
        return self.active_run

    def log_params(self, params: Dict[str, Any]):
        """Logs experiment hyperparameters."""
        # Convert any non-primitive types to str
        clean_params = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()}
        mlflow.log_params(clean_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Logs metrics at a specific epoch/step."""
        clean_metrics = {k: float(v) for k, v in metrics.items()}
        mlflow.log_metrics(clean_metrics, step=step)

    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """Logs a single artifact file."""
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path, artifact_path=artifact_path)

    def log_model(self, model: torch.nn.Module, artifact_path: str = "model", input_example: Optional[torch.Tensor] = None):
        """Logs the PyTorch model to the MLflow artifact store."""
        try:
            if input_example is None:
                input_example = torch.zeros(1, 1, 256, 256)
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                input_example=input_example.cpu().numpy()
            )
            print(f"[MLflow] Model successfully logged to artifact path: '{artifact_path}'")
        except Exception as e:
            print(f"[MLflow Warning] Failed to log PyTorch model via log_model: {e}")

    def end_run(self, status: str = "FINISHED"):
        """Ends the currently active MLflow run."""
        if mlflow.active_run():
            mlflow.end_run(status=status)
            print(f"[MLflow] Run ended with status: {status}")
