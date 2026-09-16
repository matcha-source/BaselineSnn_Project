import json
from pathlib import Path


class ResultsManager:
    """
    Stores and manages results for a single experiment.

    The complete experiment is saved in one JSON file.
    """

    def __init__(
        self,
        experiment_name,
        results_root="results"
    ):
        self.experiment_name = experiment_name

        self.file_path = (
            Path(results_root)
            / experiment_name
            / "results.json"
        )

        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.data = {
            "experiment": {},
            "training": {
                "epochs": [],
                "train_loss": [],
                "train_accuracy": [],
                "validation_loss": [],
                "validation_accuracy": [],
                "epoch_time_seconds": [],
            },
            "training_summary": {},
            "evaluation": {
                "classification_metrics": {},
                "per_class_metrics": {},
                "confusion_matrix": [],
                "performance": {},
                "model": {},
                "snn_metrics": {}
            }
        }

    def add_experiment_config(self, config):
        """
        Store experiment configuration.
        """

        self.data["experiment"] = config

    def add_training_epoch(
        self,
        epochs,
        train_loss,
        train_accuracy,
        validation_loss,
        validation_accuracy,
        epoch_time_seconds,
    ):
        """
        Add results from one training epoch.
        """

        self.data["training"]["epochs"].append(epochs)
        self.data["training"]["train_loss"].append(train_loss)
        self.data["training"]["train_accuracy"].append(train_accuracy)
        self.data["training"]["validation_loss"].append(validation_loss)
        self.data["training"]["validation_accuracy"].append(validation_accuracy)
        self.data["training"]["epoch_time_seconds"].append(epoch_time_seconds)

    def add_training_summary(
        self,
        total_training_time,
        # best_val_accuracy,
        # best_epoch
    ):
        """
        Store final training statistics.
        """

        self.data["training_summary"] = {
            "total_training_time_seconds":
                total_training_time,

            # "best_val_accuracy":
            #     best_val_accuracy,
            #
            # "best_epoch":
            #     best_epoch
        }

    def add_evaluation_results(
        self,
        classification_metrics,
        # confusion_matrix,
        # per_class_metrics=None,
        performance=None,
        model=None,
        snn_metrics=None
    ):
        """
        Store final evaluation results.
        """

        self.data["evaluation"][
            "classification_metrics"
        ] = classification_metrics

        # self.data["evaluation"][
        #     "confusion_matrix"
        # ] = confusion_matrix
        #
        # self.data["evaluation"][
        #     "per_class_metrics"
        # ] = per_class_metrics or {}

        self.data["evaluation"][
            "performance"
        ] = performance or {}

        self.data["evaluation"][
            "model"
        ] = model or {}

        self.data["evaluation"][
            "snn_metrics"
        ] = snn_metrics or {}

    def save(self):
        """
        Save the complete experiment to JSON.
        """

        with self.file_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.data,
                file,
                indent=4
            )

    def load(self):
        """
        Load experiment results from JSON.
        """

        with self.file_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            self.data = json.load(file)

        return self.data
