import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score, make_scorer, precision_recall_curve,
    average_precision_score, classification_report,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer



class KNNHateSpeechClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible KNN-based hate speech classifier using sentence-transformer embeddings.
    
    Args:
        threshold (float): Probability threshold for binary classification.
        model_name (str): SentenceTransformer model to use.
        n_neighbors (int): Number of neighbors for KNN.
        precomputed (bool): Whether input to fit/predict is already embedded.
    """

    def __init__(self, threshold=0.25, model_name='all-MiniLM-L6-v2', n_neighbors=5, precomputed=False):
        self.threshold = threshold
        self.model_name = model_name
        self.n_neighbors = n_neighbors
        self.precomputed = precomputed
        self.embedder = SentenceTransformer(model_name)
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.knn.set_params(n_neighbors=self.n_neighbors)
        if self.precomputed:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_emb = self.embedder.encode(X, show_progress_bar=False)
            X_scaled = self.scaler.fit_transform(X_emb)
        self.knn.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        if self.precomputed:
            X_scaled = self.scaler.transform(X)
        else:
            X_emb = self.embedder.encode(X, show_progress_bar=False)
            X_scaled = self.scaler.transform(X_emb)
        return self.knn.predict_proba(X_scaled)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= self.threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))

    def get_embeddings(self, texts, scale=True):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        if scale:
            return self.scaler.fit_transform(embeddings)
        return embeddings

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Hate", "Hate"],
                    yticklabels=["Not Hate", "Hate"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def show_misclassifications(self, X, y_true, y_pred):
        X = pd.Series(X)
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)
        mismatches = X[y_true != y_pred]
        print("\nFALSE POSITIVES:")
        print(mismatches[(y_true == 0) & (y_pred == 1)].head(5).to_string(index=False))
        print("\nFALSE NEGATIVES:")
        print(mismatches[(y_true == 1) & (y_pred == 0)].head(5).to_string(index=False))

    def plot_roc_curve(self, X, y_true):
        probs = self.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - KNN Hate Speech Classifier")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class LogisticHateSpeech(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible Logistic Regression-based hate speech classifier using sentence-transformer embeddings.

    Args:
        threshold (float): Probability threshold for binary classification.
        model_name (str): SentenceTransformer model to use.
        penalty (str): Penalty type for logistic regression ('l1', 'l2', or 'none').
        C (float): Inverse regularization strength.
        max_iter (int): Maximum iterations for solver.
        precomputed (bool): Whether input to fit/predict is already embedded.
    """
    def __init__(self, threshold=0.25, model_name="all-MiniLM-L6-v2",
                 penalty='l2', C=1.0, max_iter=1000, precomputed=False):
        self.threshold = threshold
        self.model_name = model_name
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.precomputed = precomputed

        self.embedder = SentenceTransformer(model_name)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(penalty=None, C=self.C, max_iter=self.max_iter)

    def fit(self, X, y):
        if self.precomputed:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_emb = self.get_embeddings(X, scale=True)
            X_scaled = X_emb
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        if self.precomputed:
            X_scaled = self.scaler.transform(X)
        else:
            X_emb = self.get_embeddings(X, scale=True)
            X_scaled = X_emb
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))

    def get_embeddings(self, texts, scale=True):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        return self.scaler.fit_transform(embeddings) if scale else embeddings



class RidgeHateSpeechClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible Ridge Logistic Regression-based hate speech classifier using sentence-transformer embeddings.

    Args:
        threshold (float): Probability threshold for binary classification.
        model_name (str): SentenceTransformer model to use.
        C (float): Inverse regularization strength.
        max_iter (int): Maximum iterations for solver.
        precomputed (bool): Whether input to fit/predict is already embedded.
    """
    def __init__(self, threshold=0.25, model_name='all-MiniLM-L6-v2',
                 C=1.0, max_iter=1000, precomputed=False):
        self.threshold = threshold
        self.model_name = model_name
        self.C = C
        self.max_iter = max_iter
        self.precomputed = precomputed

        self.embedder = SentenceTransformer(model_name)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(penalty='l2', C=self.C, max_iter=self.max_iter)

    def fit(self, X, y):
        if self.precomputed:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_emb = self.get_embeddings(X, scale=True)
            X_scaled = X_emb
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        if self.precomputed:
            X_scaled = self.scaler.transform(X)
        else:
            X_emb = self.get_embeddings(X, scale=True)
            X_scaled = X_emb
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))

    def get_embeddings(self, texts, scale=True):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        return self.scaler.fit_transform(embeddings) if scale else embeddings
    
class LassoHateSpeechClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible Lasso Logistic Regression-based hate speech classifier using sentence-transformer embeddings.

    Args:
        threshold (float): Probability threshold for binary classification.
        model_name (str): SentenceTransformer model to use.
        C (float): Inverse regularization strength.
        max_iter (int): Maximum iterations for solver.
        precomputed (bool): Whether input to fit/predict is already embedded.
    """
    def __init__(self, threshold=0.25, model_name='all-MiniLM-L6-v2',
                 C=1.0, max_iter=1000, precomputed=False):
        self.threshold = threshold
        self.model_name = model_name
        self.C = C
        self.max_iter = max_iter
        self.precomputed = precomputed

        self.embedder = SentenceTransformer(model_name)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(penalty='l1', solver='liblinear', C=self.C, max_iter=self.max_iter)

    def fit(self, X, y):
        if self.precomputed:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_emb = self.get_embeddings(X, scale=True)
            X_scaled = X_emb
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        if self.precomputed:
            X_scaled = self.scaler.transform(X)
        else:
            X_emb = self.get_embeddings(X, scale=True)
            X_scaled = X_emb
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))

    def get_embeddings(self, texts, scale=True):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        return self.scaler.fit_transform(embeddings) if scale else embeddings


class NBHateSpeechClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.25, model_name='all-MiniLM-L6-v2', precomputed=False):
        self.threshold = threshold
        self.model_name = model_name
        self.precomputed = precomputed
        self.embedder = SentenceTransformer(model_name)
        self.scaler = StandardScaler()
        self.nb = GaussianNB()  

    def fit(self, X, y):
        if self.precomputed:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_emb = self.embedder.encode(X, show_progress_bar=False)
            X_scaled = self.scaler.fit_transform(X_emb)
        self.nb.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        if self.precomputed:
            X_scaled = self.scaler.transform(X)
        else:
            X_emb = self.embedder.encode(X, show_progress_bar=False)
            X_scaled = self.scaler.transform(X_emb)
        return self.nb.predict_proba(X_scaled)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= self.threshold).astype(int)

    def get_embeddings(self, texts, scale=True):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        if scale:
            return self.scaler.fit_transform(embeddings)
        return embeddings

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Hate", "Hate"],
                    yticklabels=["Not Hate", "Hate"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def show_misclassifications(self, X, y_true, y_pred):
        X = pd.Series(X)
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)
        mismatches = X[y_true != y_pred]
        print("\nFALSE POSITIVES:")
        print(mismatches[(y_true == 0) & (y_pred == 1)].head(5).to_string(index=False))
        print("\nFALSE NEGATIVES:")
        print(mismatches[(y_true == 1) & (y_pred == 0)].head(5).to_string(index=False))


class HateSpeechRFClassifier:
    def __init__(self, threshold: float = 0.75, n_estimators: int = 100, max_depth: int = None, class_weight=None):
        self.threshold = threshold
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42
        )

    def embed(self, texts: pd.Series) -> np.ndarray:
        return self.encoder.encode(texts.tolist(), show_progress_bar=False)

    def train(self, X: pd.Series, y: pd.Series) -> None:
        X_embed = self.embed(X)
        self.model.fit(X_embed, y)

    def predict_proba(self, X: pd.Series) -> pd.Series:
        X_embed = self.embed(X)
        proba = self.model.predict_proba(X_embed)[:, 1]
        return pd.Series(proba, index=X.index)

    def predict(self, X: pd.Series) -> pd.Series:
        proba = self.predict_proba(X)
        return (proba > self.threshold).astype(int)

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> None:
        y_pred = self.predict(X_test)
        print(classification_report(y_test, y_pred))

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Hate", "Hate"], yticklabels=["Not Hate", "Hate"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def show_misclassifications(self, X: pd.Series, y_true: pd.Series, y_pred: pd.Series):
        mismatches = X[(y_true != y_pred)]
        print("\nFALSE POSITIVES:")
        print(mismatches[(y_true == 0) & (y_pred == 1)].head(5).to_string(index=False))

        print("\nFALSE NEGATIVES:")
        print(mismatches[(y_true == 1) & (y_pred == 0)].head(5).to_string(index=False))

    
class HateSpeechXGBClassifier:
    def __init__(self, threshold: float = 0.75, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.3):
        self.threshold = threshold
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            eval_metric='logloss'
        )

    def embed(self, texts: pd.Series) -> np.ndarray:
        return self.encoder.encode(texts.tolist(), show_progress_bar=False)

    def train(self, X: pd.Series, y: pd.Series) -> None:
        X_embed = self.embed(X)
        self.model.fit(X_embed, y)

    def predict_proba(self, X: pd.Series) -> pd.Series:
        X_embed = self.embed(X)
        proba = self.model.predict_proba(X_embed)[:, 1]
        return pd.Series(proba, index=X.index)

    def predict(self, X: pd.Series) -> pd.Series:
        proba = self.predict_proba(X)
        return (proba > self.threshold).astype(int)

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> None:
        y_pred = self.predict(X_test)
        print(classification_report(y_test, y_pred))

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Hate", "Hate"], yticklabels=["Not Hate", "Hate"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def show_misclassifications(self, X: pd.Series, y_true: pd.Series, y_pred: pd.Series):
        mismatches = X[(y_true != y_pred)]
        print("\nFALSE POSITIVES:")
        print(mismatches[(y_true == 0) & (y_pred == 1)].head(5).to_string(index=False))

        print("\nFALSE NEGATIVES:")
        print(mismatches[(y_true == 1) & (y_pred == 0)].head(5).to_string(index=False))
