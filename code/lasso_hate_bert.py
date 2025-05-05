from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models

class LassoHateBERT(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression hate speech classifier using SentenceTransformer or HateBERT.

    Args:
        threshold (float): Classification threshold.
        model_name (str): Name of Sentence-BERT model.
        precomputed (bool): If True, input X is already embedded.
        use_hatebert (bool): If True, use HateBERT instead of SentenceTransformer.
        penalty (str or None): LogisticRegression penalty ('l2', 'l1', 'elasticnet', or None).
    """

    def __init__(self, threshold=0.25, model_name='all-MiniLM-L6-v2',
                 precomputed=False, use_hatebert=False,
                 penalty="l1", C=1.0, solver='lbfgs', random_state=42):
        self.threshold = threshold
        self.model_name = model_name
        self.precomputed = precomputed
        self.use_hatebert = use_hatebert
        self.penalty = None if penalty == 'none' else penalty
        self.C = C
        self.solver = solver
        self.random_state = random_state

        self.embedder = None
        self.scaler = StandardScaler()
        self.model_ = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
            max_iter=1000,
            random_state=self.random_state
        )

    def _load_embedder(self):
        if self.embedder is None:
            if self.use_hatebert:
                word_model = models.Transformer("GroNLP/hateBERT")
                pooling_model = models.Pooling(word_model.get_word_embedding_dimension())
                self.embedder = SentenceTransformer(modules=[word_model, pooling_model])
            else:
                self.embedder = SentenceTransformer(self.model_name)

    def fit(self, X, y):
        if not self.precomputed:
            self._load_embedder()
            X = self.get_embeddings(X)
        X_scaled = self.scaler.fit_transform(X)
        self.model_.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        if not self.precomputed:
            self._load_embedder()
            X = self.get_embeddings(X)
        X_scaled = self.scaler.transform(X)
        return self.model_.predict_proba(X_scaled)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= self.threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))

    def get_embeddings(self, texts, batch_size=32):
        self._load_embedder()
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        return self.embedder.encode(
            texts, batch_size=batch_size,
            show_progress_bar=False, convert_to_numpy=True
        )

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Hate", "Hate"],
                    yticklabels=["Not Hate", "Hate"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='jpg')
        plt.close()


    def plot_roc_curve(self, X, y_true, save_path=None):
        probs = self.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Logistic Hate Speech Classifier")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='jpg')
        plt.close()


train_df = pd.read_csv(r"C:\Users\nicol\Desktop\Ed_Krueger\eco395m_ml_final\data\train_data.csv")
test_df = pd.read_csv(r"C:\Users\nicol\Desktop\Ed_Krueger\eco395m_ml_final\data\test_data.csv", sep=";")

train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()
test_texts = test_df["comment"].tolist()
test_labels = test_df["isHate"].tolist()

model = LassoHateBERT(threshold= 0.25, use_hatebert=True, precomputed= True)
X_train = model.get_embeddings(train_texts)
X_test = model.get_embeddings(test_texts)
pca = PCA(n_components=50)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

model.fit(X_train_reduced, train_labels)

test_labels_bin = np.array(test_labels).astype(int)  
probs = model.predict_proba(X_test_reduced)[:, 1]
custom_preds = (probs >= 0.25).astype(int)

print(classification_report(test_labels_bin, custom_preds))

model.plot_confusion_matrix(test_labels_bin, custom_preds,
    save_path=r"C:\Users\nicol\Desktop\Ed_Krueger\eco395m_ml_final\images\lasso_hate_bert_confusion_matrix.jpg")

model.plot_roc_curve(X_test_reduced, test_labels_bin,
    save_path=r"C:\Users\nicol\Desktop\Ed_Krueger\eco395m_ml_final\images\lasso_hate_bert_roc_curve.jpg")