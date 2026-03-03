import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import optuna
from dotenv import load_dotenv
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score

load_dotenv()

METRIC = os.getenv("METRIC", "accuracy")

SKLEARN_SCORING = {
    "accuracy": "accuracy",
    "f1": "f1_weighted",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "roc_auc": "roc_auc",
    "r2": "r2",
    "neg_mean_squared_error": "neg_mean_squared_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
    "neg_log_loss": "neg_log_loss",
}

METRIC_DIRECTION = {
    "accuracy": "maximize",
    "f1": "maximize",
    "precision": "maximize",
    "recall": "maximize",
    "roc_auc": "maximize",
    "r2": "maximize",
    "neg_mean_squared_error": "minimize",
    "neg_mean_absolute_error": "minimize",
    "neg_log_loss": "minimize",
}


class BaseModel(ABC):
    def __init__(self, data_dir: str = "data/processed", n_trials: int = 50):
        self.data_dir = Path(data_dir)
        self.model = None
        self.best_params: dict | None = None
        self.n_trials = n_trials
        self.metric = METRIC
        self.scoring = SKLEARN_SCORING.get(self.metric, self.metric)
        self.direction = METRIC_DIRECTION.get(self.metric, "maximize")

    def load_data(self):
        X_train = np.load(self.data_dir / "X_train_preprocessed.npy", allow_pickle=True)
        X_test = np.load(self.data_dir / "X_test_preprocessed.npy", allow_pickle=True)
        y_train = np.load(self.data_dir / "y_train.npy", allow_pickle=True)
        y_test = np.load(self.data_dir / "y_test.npy", allow_pickle=True)
        return X_train, X_test, y_train, y_test

    @abstractmethod
    def build_model(self, params: dict | None = None):
        """Return an unfitted estimator, optionally configured with hyperparams."""
        ...

    @abstractmethod
    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Define the Optuna search space and return a dict of suggested hyperparams."""
        ...

    def _objective(self, trial: optuna.Trial, X_train, y_train):
        params = self.suggest_params(trial)
        model = self.build_model(params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=self.scoring)
        return scores.mean()

    def optimize(self, X_train, y_train):
        study = optuna.create_study(direction=self.direction)
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=self.n_trials,
        )
        self.best_params = study.best_params
        print(f"Best {self.metric} (CV): {study.best_value:.4f}")
        print(f"Best params: {self.best_params}")
        return study

    def train(self, X_train, y_train):
        self.model = self.build_model(self.best_params)
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred) -> dict:
        scorer = get_scorer(self.scoring)
        score = scorer._score_func(y_true, y_pred, **scorer._kwargs)
        return {self.metric: score}

    def run(self):
        X_train, X_test, y_train, y_test = self.load_data()
        self.optimize(X_train, y_train)
        self.train(X_train, y_train)
        y_pred = self.predict(X_test)
        metrics = self.evaluate(y_test, y_pred)
        for name, value in metrics.items():
            print(f"Test {name}: {value:.4f}")
        return metrics
