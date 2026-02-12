"""
Hybrid FinBERT-VADER Sentiment Analysis with Finance-Aware Routing

Multi-Tiered Ensemble approach for financial sentiment analysis:
- VADER (fast lexicon) for high-confidence samples
- FinBERT (transformer) for ambiguous cases
- Finance-aware Gate MLP router with calibrated uncertainty,
  agreement-aware supervision, and linguistic features
- Cost-aware threshold optimization

Hypothesis: Achieves >=95% of FinBERT accuracy at <=20% computational cost.
"""

import json
import os
import random
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import optimize, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
DATA_DIR = Path("FinancialPhraseBank-v1.0")

LABEL_LIST = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

# ============================================================
# Data Loading with Agreement Metadata
# ============================================================


def load_single_phrasebank(filepath: str, encoding: str = "latin-1") -> Dict[str, str]:
    """Load a single PhraseBank file. Returns {sentence: label}."""
    data = {}
    with open(filepath, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if "@" in line:
                parts = line.rsplit("@", 1)
                if len(parts) == 2:
                    text, label = parts[0].strip(), parts[1].strip()
                    data[text] = label
    return data


def load_all_phrasebank(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load all 4 agreement levels and derive ambiguity metadata.
    Ambiguity score: 0.0 (in AllAgree = easy) to 1.0 (only in 50Agree = hard).
    """
    files = {
        "50": os.path.join(data_dir, "Sentences_50Agree.txt"),
        "66": os.path.join(data_dir, "Sentences_66Agree.txt"),
        "75": os.path.join(data_dir, "Sentences_75Agree.txt"),
        "100": os.path.join(data_dir, "Sentences_AllAgree.txt"),
    }

    datasets = {}
    for level, fp in files.items():
        if os.path.exists(fp):
            datasets[level] = load_single_phrasebank(fp)
        else:
            print(f"Warning: {fp} not found")

    all_sentences = datasets.get("50", {})
    if not all_sentences:
        raise FileNotFoundError("Sentences_50Agree.txt not found")

    rows = []
    for text, label in all_sentences.items():
        in_100 = text in datasets.get("100", {})
        in_75 = text in datasets.get("75", {})
        in_66 = text in datasets.get("66", {})

        if in_100:
            tier = "100"
            ambiguity = 0.0
        elif in_75:
            tier = "75"
            ambiguity = 0.33
        elif in_66:
            tier = "66"
            ambiguity = 0.67
        else:
            tier = "50"
            ambiguity = 1.0

        rows.append(
            {
                "text": text,
                "label": label,
                "agreement_tier": tier,
                "ambiguity_score": ambiguity,
            }
        )

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} sentences from FinancialPhraseBank")
    print(f"Agreement tiers: {df['agreement_tier'].value_counts().to_dict()}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df


def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.2, seed=SEED):
    """Stratified train/val/test split."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_ratio, random_state=seed, stratify=train_df["label"]
    )
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ============================================================
# Baseline Models
# ============================================================


@dataclass
class SentimentResult:
    """Container for sentiment prediction results."""

    text: str
    predicted_label: str
    confidence: float
    model_used: str
    inference_time: float


class VADERAnalyzer:
    """VADER sentiment analyzer with feature extraction."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Returns (label, confidence, raw_scores)."""
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        confidence = abs(compound)
        return label, confidence, scores

    def analyze_batch(self, texts: List[str]) -> List[Tuple[str, float, Dict]]:
        return [self.analyze(t) for t in texts]

    def get_features(self, text: str) -> np.ndarray:
        """6-dim feature vector from VADER scores."""
        scores = self.analyzer.polarity_scores(text)
        return np.array(
            [
                scores["pos"],
                scores["neg"],
                scores["neu"],
                scores["compound"],
                abs(scores["compound"]),
                len(text.split()),
            ]
        )

    def get_features_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.get_features(t) for t in texts])


class FinBERTAnalyzer:
    """FinBERT sentiment analyzer returning logits, probs, and labels."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.device = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self._build_label_map()

    def _build_label_map(self):
        """Build label map from model config, with fallback."""
        if hasattr(self.model.config, "id2label"):
            config_map = self.model.config.id2label
            self.label_map = {}
            for idx, lbl in config_map.items():
                self.label_map[int(idx)] = lbl.lower()
            print(f"FinBERT label map from config: {self.label_map}")
        else:
            self.label_map = {0: "positive", 1: "negative", 2: "neutral"}
            print(f"FinBERT using fallback label map: {self.label_map}")

    def analyze(self, text: str) -> Tuple[str, float]:
        """Returns (label, confidence)."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
        return self.label_map[predicted.item()], confidence.item()

    def analyze_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Tuple[str, float]]:
        """Batch inference returning (label, confidence) pairs."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                confidences, predictions = torch.max(probs, dim=1)
            for conf, pred in zip(confidences.tolist(), predictions.tolist()):
                results.append((self.label_map[pred], conf))
        return results

    def get_logits_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch inference returning (logits, probs) as numpy arrays."""
        all_logits = []
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_logits), np.concatenate(all_probs)


class TextBlobAnalyzer:
    """TextBlob baseline sentiment analyzer."""

    def analyze(self, text: str) -> Tuple[str, float]:
        from textblob import TextBlob

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.05:
            label = "positive"
        elif polarity < -0.05:
            label = "negative"
        else:
            label = "neutral"
        return label, abs(polarity)

    def analyze_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        return [self.analyze(t) for t in texts]


# ============================================================
# Temperature Scaling (Calibration)
# ============================================================


class TemperatureScaler:
    """Post-hoc temperature scaling for calibrating FinBERT probabilities."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        """Fit temperature on validation logits and labels."""
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        def nll_loss(t):
            t = max(t, 0.01)
            scaled = logits_tensor / t
            log_probs = torch.log_softmax(scaled, dim=1)
            loss = nn.NLLLoss()(log_probs, labels_tensor)
            return loss.item()

        # Wider bound — FinBERT can be very overconfident
        result = optimize.minimize_scalar(
            nll_loss, bounds=(0.1, 50.0), method="bounded"
        )
        self.temperature = result.x
        print(f"Fitted temperature: {self.temperature:.4f}")

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits, return calibrated probabilities."""
        scaled = torch.tensor(logits, dtype=torch.float32) / self.temperature
        probs = torch.softmax(scaled, dim=1).numpy()
        return probs

    def get_uncertainty_features(self, calibrated_probs: np.ndarray) -> np.ndarray:
        """Returns [max_prob, negative_entropy] per sample."""
        max_prob = np.max(calibrated_probs, axis=1)
        entropy = -np.sum(calibrated_probs * np.log(calibrated_probs + 1e-10), axis=1)
        neg_entropy = -entropy
        return np.column_stack([max_prob, neg_entropy])


# ============================================================
# Linguistic Feature Extractor
# ============================================================


class LinguisticFeatureExtractor:
    """Extract finance-aware linguistic features from text."""

    MODAL_VERBS = {
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "will",
        "must",
    }
    FORWARD_LOOKING = {
        "expect",
        "expects",
        "expected",
        "expecting",
        "anticipate",
        "anticipates",
        "anticipated",
        "forecast",
        "forecasts",
        "forecasted",
        "outlook",
        "guidance",
        "projection",
        "projections",
        "estimate",
        "estimates",
        "estimated",
        "predict",
        "predicts",
        "predicted",
        "plan",
        "plans",
        "planned",
    }
    DIRECTIONAL_VERBS = {
        "rise",
        "rises",
        "rose",
        "risen",
        "rising",
        "fall",
        "falls",
        "fell",
        "fallen",
        "falling",
        "increase",
        "increases",
        "increased",
        "increasing",
        "decrease",
        "decreases",
        "decreased",
        "decreasing",
        "grow",
        "grows",
        "grew",
        "grown",
        "growing",
        "decline",
        "declines",
        "declined",
        "declining",
        "surge",
        "surges",
        "surged",
        "surging",
        "drop",
        "drops",
        "dropped",
        "dropping",
        "gain",
        "gains",
        "gained",
        "gaining",
        "lose",
        "loses",
        "lost",
        "losing",
        "climb",
        "climbs",
        "climbed",
        "climbing",
        "slip",
        "slips",
        "slipped",
        "slipping",
        "improve",
        "improves",
        "improved",
        "improving",
        "worsen",
        "worsens",
        "worsened",
        "worsening",
    }
    NEGATIONS = {
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "hardly",
        "barely",
        "scarcely",
        "n't",
    }
    CONTRAST_WORDS = {
        "but",
        "however",
        "although",
        "though",
        "despite",
        "nevertheless",
        "nonetheless",
        "yet",
        "while",
        "whereas",
        "conversely",
        "instead",
    }

    def extract(self, text: str) -> np.ndarray:
        """8-dimensional feature vector."""
        words = text.lower().split()
        word_set = set(words)

        modal_count = len(word_set & self.MODAL_VERBS)
        forward_count = len(word_set & self.FORWARD_LOOKING)
        directional_count = len(word_set & self.DIRECTIONAL_VERBS)
        negation_count = sum(
            1 for w in words if w in self.NEGATIONS or w.endswith("n't")
        )
        number_count = len(re.findall(r"\b\d+[\d,.]*\b", text))
        contrast_count = len(word_set & self.CONTRAST_WORDS)
        word_count = len(words)
        has_question = 1.0 if "?" in text else 0.0

        return np.array(
            [
                modal_count,
                forward_count,
                directional_count,
                negation_count,
                number_count,
                contrast_count,
                word_count,
                has_question,
            ],
            dtype=np.float32,
        )

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.extract(t) for t in texts])


# ============================================================
# Ambiguity Predictor
# ============================================================


class AmbiguityPredictor:
    """Predicts ambiguity score from text using sentence embeddings."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self._sentence_model = None

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer

            self._sentence_model = SentenceTransformer(
                "sentence-transformers/paraphrase-MiniLM-L6-v2", device=str(DEVICE)
            )
        return self._sentence_model.encode(
            texts, show_progress_bar=False, batch_size=64
        )

    def train(self, texts: List[str], ambiguity_scores: np.ndarray):
        """Train logistic regression to predict ambiguity."""
        print("Training ambiguity predictor...")
        embeddings = self._get_embeddings(texts)
        X = self.scaler.fit_transform(embeddings)
        y_binary = (ambiguity_scores > 0.5).astype(int)
        self.model = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        self.model.fit(X, y_binary)
        train_acc = self.model.score(X, y_binary)
        print(f"Ambiguity predictor train accuracy: {train_acc:.4f}")

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict ambiguity probability for each text."""
        if self.model is None:
            raise ValueError("Model not trained")
        embeddings = self._get_embeddings(texts)
        X = self.scaler.transform(embeddings)
        return self.model.predict_proba(X)[:, 1]


# ============================================================
# Gate MLP Router
# ============================================================


class GateMLP(nn.Module):
    """Small MLP for routing: maps features to scalar routing score."""

    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GateMLPRouter:
    """
    Finance-aware routing gate.

    At TRAINING time: uses VADER features + FinBERT calibrated uncertainty +
    ambiguity + linguistic features to learn the routing decision.

    At INFERENCE time: uses only fast features (VADER + linguistic + ambiguity)
    so we don't need to run FinBERT before the routing decision.

    The gate outputs: high score = VADER likely correct (accept student),
                      low score  = VADER likely wrong (defer to teacher).
    """

    def __init__(self, input_dim: int = 15, lr: float = 1e-3, epochs: int = 100):
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.gate = GateMLP(input_dim).to(DEVICE)
        self.feature_scaler = StandardScaler()
        self.threshold = 0.5
        # For training with extra features
        self.gate_train = None
        self.train_input_dim = None

    def _prepare_inference_features(
        self,
        vader_features: np.ndarray,
        ambiguity_scores: np.ndarray,
        linguistic_features: np.ndarray,
    ) -> np.ndarray:
        """Concatenate inference-time features (no FinBERT needed)."""
        ambiguity_col = ambiguity_scores.reshape(-1, 1)
        return np.concatenate(
            [vader_features, ambiguity_col, linguistic_features], axis=1
        )

    def _prepare_train_features(
        self,
        vader_features: np.ndarray,
        uncertainty_features: np.ndarray,
        ambiguity_scores: np.ndarray,
        linguistic_features: np.ndarray,
    ) -> np.ndarray:
        """Concatenate all features including FinBERT uncertainty (training only)."""
        ambiguity_col = ambiguity_scores.reshape(-1, 1)
        return np.concatenate(
            [vader_features, uncertainty_features, ambiguity_col, linguistic_features],
            axis=1,
        )

    def train(
        self,
        vader_features: np.ndarray,
        ambiguity_scores: np.ndarray,
        linguistic_features: np.ndarray,
        vader_correct: np.ndarray,
    ):
        """Train the gate MLP using inference-time features only."""
        features = self._prepare_inference_features(
            vader_features, ambiguity_scores, linguistic_features
        )
        features = self.feature_scaler.fit_transform(features)

        X = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(vader_correct, dtype=torch.float32).to(DEVICE)

        # Class-weighted BCE to handle imbalanced VADER correctness
        n_correct = (y == 1).sum().item()
        n_wrong = (y == 0).sum().item()
        pos_weight_val = n_wrong / max(n_correct, 1.0)
        print(
            f"  VADER correct: {n_correct}, wrong: {n_wrong}, pos_weight: {pos_weight_val:.2f}"
        )

        # Use BCEWithLogitsLoss with pos_weight for better training
        # But our model already has sigmoid, so use weighted BCE manually
        criterion = nn.BCELoss(reduction="none")
        optimizer = optim.Adam(self.gate.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Weights: upweight the minority class (VADER wrong samples)
        weights = torch.where(
            y == 1,
            torch.tensor(1.0).to(DEVICE),
            torch.tensor(pos_weight_val).to(DEVICE),
        )

        best_loss = float("inf")
        self.gate.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            scores = self.gate(X)
            loss_per_sample = criterion(scores, y)
            loss = (loss_per_sample * weights).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

        self.gate.eval()
        with torch.no_grad():
            train_scores = self.gate(X).cpu().numpy()
            train_preds = (train_scores >= 0.5).astype(int)
            train_acc = accuracy_score(vader_correct, train_preds)
            # Check score distribution
            print(
                f"  Gate scores: min={train_scores.min():.3f}, max={train_scores.max():.3f}, "
                f"mean={train_scores.mean():.3f}, std={train_scores.std():.3f}"
            )
        print(f"  Gate MLP train accuracy: {train_acc:.4f}")

    def predict(
        self,
        vader_features: np.ndarray,
        ambiguity_scores: np.ndarray,
        linguistic_features: np.ndarray,
    ) -> np.ndarray:
        """Predict routing scores (higher = use VADER). Uses only fast features."""
        features = self._prepare_inference_features(
            vader_features, ambiguity_scores, linguistic_features
        )
        features = self.feature_scaler.transform(features)
        X = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        self.gate.eval()
        with torch.no_grad():
            scores = self.gate(X).cpu().numpy()
        return scores


# ============================================================
# Cost-Aware Threshold Optimizer
# ============================================================


class CostAwareOptimizer:
    """
    Optimizes routing threshold tau on validation set.

    The cost function is: R(tau) = error_rate + lambda * finbert_usage_fraction

    IMPORTANT: lambda must be calibrated to the actual cost ratio.
    If FinBERT takes ~50x longer than VADER, then lambda should reflect
    the marginal accuracy cost of using FinBERT, not a raw penalty.

    We use smaller lambda values (0.01 - 0.5) to allow meaningful FinBERT usage.
    """

    def __init__(self, lambda_values=None, tau_grid=None):
        self.lambda_values = lambda_values or [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
        self.tau_grid = tau_grid or np.arange(0.01, 1.00, 0.01)
        self.results = {}

    def optimize(
        self,
        routing_scores: np.ndarray,
        vader_preds: np.ndarray,
        finbert_preds: np.ndarray,
        true_labels: np.ndarray,
    ) -> Dict:
        """Find optimal tau for each lambda value."""
        best_results = {}

        for lam in self.lambda_values:
            best_tau = 0.5
            best_cost = float("inf")

            for tau in self.tau_grid:
                use_vader = routing_scores >= tau
                preds = np.where(use_vader, vader_preds, finbert_preds)
                error_rate = 1.0 - accuracy_score(true_labels, preds)
                finbert_fraction = (~use_vader).mean()
                cost = error_rate + lam * finbert_fraction

                if cost < best_cost:
                    best_cost = cost
                    best_tau = tau

            use_vader = routing_scores >= best_tau
            preds = np.where(use_vader, vader_preds, finbert_preds)
            acc = accuracy_score(true_labels, preds)
            f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
            finbert_pct = (~use_vader).mean() * 100

            best_results[lam] = {
                "tau": best_tau,
                "accuracy": acc,
                "f1_macro": f1,
                "finbert_pct": finbert_pct,
                "vader_pct": use_vader.mean() * 100,
                "cost": best_cost,
            }

        self.results = best_results
        return best_results

    def get_pareto_frontier(
        self,
        routing_scores: np.ndarray,
        vader_preds: np.ndarray,
        finbert_preds: np.ndarray,
        true_labels: np.ndarray,
    ) -> List[Dict]:
        """Compute accuracy vs. cost trade-off across all thresholds."""
        frontier = []
        for tau in self.tau_grid:
            use_vader = routing_scores >= tau
            preds = np.where(use_vader, vader_preds, finbert_preds)
            acc = accuracy_score(true_labels, preds)
            f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
            finbert_pct = (~use_vader).mean() * 100
            frontier.append(
                {
                    "tau": float(tau),
                    "accuracy": acc,
                    "f1_macro": f1,
                    "finbert_pct": finbert_pct,
                    "vader_pct": use_vader.mean() * 100,
                }
            )
        return frontier

    def find_target_operating_point(
        self,
        routing_scores: np.ndarray,
        vader_preds: np.ndarray,
        finbert_preds: np.ndarray,
        true_labels: np.ndarray,
        finbert_acc: float,
        target_retention: float = 0.95,
        max_finbert_pct: float = 20.0,
    ) -> Dict:
        """
        Find tau that achieves target accuracy retention with minimal FinBERT usage,
        or the best accuracy achievable within the cost budget.
        """
        target_acc = finbert_acc * target_retention
        best_point = None
        best_finbert_pct = 100.0

        for tau in self.tau_grid:
            use_vader = routing_scores >= tau
            preds = np.where(use_vader, vader_preds, finbert_preds)
            acc = accuracy_score(true_labels, preds)
            finbert_pct = (~use_vader).mean() * 100

            if acc >= target_acc and finbert_pct <= max_finbert_pct:
                if finbert_pct < best_finbert_pct:
                    best_finbert_pct = finbert_pct
                    best_point = {
                        "tau": float(tau),
                        "accuracy": acc,
                        "finbert_pct": finbert_pct,
                        "meets_target": True,
                    }

        # If no point meets both criteria, find best accuracy within cost budget
        if best_point is None:
            best_acc = 0
            for tau in self.tau_grid:
                use_vader = routing_scores >= tau
                finbert_pct = (~use_vader).mean() * 100
                if finbert_pct <= max_finbert_pct:
                    preds = np.where(use_vader, vader_preds, finbert_preds)
                    acc = accuracy_score(true_labels, preds)
                    if acc > best_acc:
                        best_acc = acc
                        best_point = {
                            "tau": float(tau),
                            "accuracy": acc,
                            "finbert_pct": finbert_pct,
                            "meets_target": False,
                        }

        return best_point


# ============================================================
# Hybrid Sentiment Analyzer (Orchestrator)
# ============================================================


class HybridSentimentAnalyzer:
    """
    Full hybrid pipeline:
    1. VADER fast-path (student)
    2. FinBERT teacher (deferred)
    3. Finance-aware Gate MLP router (uses only fast features at inference)
    4. Cost-aware threshold optimization

    Key design: At inference time, the router does NOT need FinBERT features.
    It uses VADER scores + linguistic features + ambiguity prediction only.
    FinBERT is only called for samples that the gate defers.
    """

    def __init__(self):
        print("Initializing models...")
        self.vader = VADERAnalyzer()
        self.finbert = FinBERTAnalyzer()
        self.textblob = TextBlobAnalyzer()
        self.temp_scaler = TemperatureScaler()
        self.ling_extractor = LinguisticFeatureExtractor()
        self.ambiguity_predictor = AmbiguityPredictor()
        # input_dim = 6 (VADER) + 1 (ambiguity) + 8 (linguistic) = 15
        self.gate_router = GateMLPRouter(input_dim=15)
        self.cost_optimizer = CostAwareOptimizer()
        self.optimal_tau = 0.5

    def train_pipeline(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train all components of the pipeline."""
        train_texts = train_df["text"].tolist()
        train_labels = train_df["label"].tolist()
        val_texts = val_df["text"].tolist()
        val_labels = val_df["label"].tolist()

        # Step 1: VADER predictions and features
        print("\n--- Step 1: VADER predictions ---")
        train_vader = self.vader.analyze_batch(train_texts)
        train_vader_preds = np.array([r[0] for r in train_vader])
        train_vader_features = self.vader.get_features_batch(train_texts)
        vader_correct = (train_vader_preds == np.array(train_labels)).astype(float)
        print(f"VADER train accuracy: {vader_correct.mean():.4f}")

        val_vader = self.vader.analyze_batch(val_texts)
        val_vader_preds = np.array([r[0] for r in val_vader])
        val_vader_features = self.vader.get_features_batch(val_texts)

        # Step 2: FinBERT calibration (for evaluation, not routing)
        print("\n--- Step 2: FinBERT logits + calibration ---")
        val_logits, val_probs = self.finbert.get_logits_batch(val_texts)
        val_label_ids = np.array([LABEL2ID[l] for l in val_labels])
        self.temp_scaler.fit(val_logits, val_label_ids)

        # Step 3: Linguistic features
        print("\n--- Step 3: Linguistic features ---")
        train_ling = self.ling_extractor.extract_batch(train_texts)
        val_ling = self.ling_extractor.extract_batch(val_texts)

        # Step 4: Ambiguity predictor
        print("\n--- Step 4: Ambiguity predictor ---")
        train_ambiguity_true = train_df["ambiguity_score"].values
        self.ambiguity_predictor.train(train_texts, train_ambiguity_true)
        train_ambiguity_pred = self.ambiguity_predictor.predict(train_texts)
        val_ambiguity_pred = self.ambiguity_predictor.predict(val_texts)

        # Step 5: Train Gate MLP (using only fast features)
        print("\n--- Step 5: Gate MLP ---")
        self.gate_router.train(
            train_vader_features,
            train_ambiguity_pred,
            train_ling,
            vader_correct,
        )

        # Step 6: Cost-aware threshold optimization on val set
        print("\n--- Step 6: Cost-aware threshold optimization ---")
        val_routing_scores = self.gate_router.predict(
            val_vader_features,
            val_ambiguity_pred,
            val_ling,
        )
        val_finbert_results = self.finbert.analyze_batch(val_texts)
        val_finbert_preds = np.array([r[0] for r in val_finbert_results])
        val_labels_arr = np.array(val_labels)

        # Get FinBERT validation accuracy for target computation
        val_finbert_acc = accuracy_score(val_labels_arr, val_finbert_preds)
        print(f"  Val FinBERT accuracy: {val_finbert_acc:.4f}")

        opt_results = self.cost_optimizer.optimize(
            val_routing_scores,
            val_vader_preds,
            val_finbert_preds,
            val_labels_arr,
        )

        # Print all lambda results
        print("\n  Cost-aware optimization results:")
        print(f"  {'λ':>6} {'τ':>6} {'Accuracy':>10} {'FinBERT%':>10}")
        print("  " + "-" * 35)
        for lam, res in sorted(opt_results.items()):
            print(
                f"  {lam:>6.2f} {res['tau']:>6.2f} {res['accuracy']:>10.4f} {res['finbert_pct']:>10.1f}%"
            )

        # Find multiple operating points for analysis
        # 1. Best accuracy within 20% FinBERT budget
        cost_target = self.cost_optimizer.find_target_operating_point(
            val_routing_scores,
            val_vader_preds,
            val_finbert_preds,
            val_labels_arr,
            val_finbert_acc,
            target_retention=0.95,
            max_finbert_pct=20.0,
        )
        # 2. Find tau that achieves 95% accuracy retention (any FinBERT %)
        acc_target = self.cost_optimizer.find_target_operating_point(
            val_routing_scores,
            val_vader_preds,
            val_finbert_preds,
            val_labels_arr,
            val_finbert_acc,
            target_retention=0.95,
            max_finbert_pct=100.0,
        )

        print(f"\n  Operating points:")
        if cost_target:
            print(
                f"  [Cost target <=20%] tau={cost_target['tau']:.3f}, "
                f"acc={cost_target['accuracy']:.4f}, "
                f"FinBERT={cost_target['finbert_pct']:.1f}%, "
                f"meets_95pct={cost_target['meets_target']}"
            )
        if acc_target:
            print(
                f"  [Acc target >=95%]  tau={acc_target['tau']:.3f}, "
                f"acc={acc_target['accuracy']:.4f}, "
                f"FinBERT={acc_target['finbert_pct']:.1f}%, "
                f"meets_95pct={acc_target['meets_target']}"
            )

        # Use the accuracy-target point as default
        if acc_target and acc_target["meets_target"]:
            self.optimal_tau = acc_target["tau"]
            print(f"\n  Selected: accuracy-target tau={self.optimal_tau:.3f}")
        elif cost_target:
            self.optimal_tau = cost_target["tau"]
            print(f"\n  Selected: cost-target tau={self.optimal_tau:.3f}")
        else:
            self.optimal_tau = opt_results[0.05]["tau"]
            print(f"\n  Selected: lambda=0.05 fallback tau={self.optimal_tau:.3f}")

        return opt_results

    def predict(
        self, texts: List[str], tau: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Run the full hybrid pipeline on texts.
        Returns (predictions, use_vader_mask, timing_info).

        Key: router only uses fast features (VADER + linguistic + ambiguity).
        FinBERT is called ONLY for deferred samples.
        """
        if tau is None:
            tau = self.optimal_tau

        timing = {}

        # VADER (fast)
        t0 = time.time()
        vader_results = self.vader.analyze_batch(texts)
        vader_preds = np.array([r[0] for r in vader_results])
        vader_features = self.vader.get_features_batch(texts)
        timing["vader"] = time.time() - t0

        # Linguistic features + ambiguity (fast)
        t0 = time.time()
        ling = self.ling_extractor.extract_batch(texts)
        ambiguity = self.ambiguity_predictor.predict(texts)
        timing["features"] = time.time() - t0

        # Gate routing (fast - no FinBERT needed)
        t0 = time.time()
        routing_scores = self.gate_router.predict(vader_features, ambiguity, ling)
        use_vader = routing_scores >= tau
        timing["routing"] = time.time() - t0

        # FinBERT ONLY for deferred samples
        t0 = time.time()
        finbert_indices = np.where(~use_vader)[0]
        finbert_preds = np.empty(len(texts), dtype=object)
        if len(finbert_indices) > 0:
            deferred_texts = [texts[i] for i in finbert_indices]
            deferred_results = self.finbert.analyze_batch(deferred_texts)
            for idx, (label, _) in zip(finbert_indices, deferred_results):
                finbert_preds[idx] = label
        timing["finbert_deferred"] = time.time() - t0

        # Combine predictions
        predictions = np.where(use_vader, vader_preds, finbert_preds)
        timing["vader_pct"] = use_vader.mean() * 100
        timing["finbert_pct"] = (~use_vader).mean() * 100
        timing["total"] = sum(
            timing[k] for k in ["vader", "features", "routing", "finbert_deferred"]
        )

        return predictions, use_vader, timing

    def predict_naive(
        self, texts: List[str], threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Naive confidence-based routing (VADER compound only, no finance-aware features)."""
        vader_results = self.vader.analyze_batch(texts)
        vader_preds = np.array([r[0] for r in vader_results])
        vader_confs = np.array([r[1] for r in vader_results])

        use_vader = vader_confs >= threshold
        finbert_indices = np.where(~use_vader)[0]
        finbert_preds = np.empty(len(texts), dtype=object)
        if len(finbert_indices) > 0:
            deferred_texts = [texts[i] for i in finbert_indices]
            deferred_results = self.finbert.analyze_batch(deferred_texts)
            for idx, (label, _) in zip(finbert_indices, deferred_results):
                finbert_preds[idx] = label

        predictions = np.where(use_vader, vader_preds, finbert_preds)
        return predictions, use_vader


# ============================================================
# Experiments
# ============================================================


def experiment_1_baselines(
    analyzer: HybridSentimentAnalyzer, test_df: pd.DataFrame
) -> Dict:
    """Experiment 1: Individual model baselines."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Individual Model Baselines")
    print("=" * 70)

    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()
    results = {}

    # VADER
    t0 = time.time()
    vader_results = analyzer.vader.analyze_batch(texts)
    vader_time = time.time() - t0
    vader_preds = [r[0] for r in vader_results]
    results["VADER"] = {
        "accuracy": accuracy_score(labels, vader_preds),
        "f1_macro": f1_score(labels, vader_preds, average="macro", zero_division=0),
        "total_time": vader_time,
        "avg_time_ms": vader_time / len(texts) * 1000,
        "report": classification_report(
            labels, vader_preds, output_dict=True, zero_division=0
        ),
    }

    # FinBERT
    t0 = time.time()
    finbert_results = analyzer.finbert.analyze_batch(texts)
    finbert_time = time.time() - t0
    finbert_preds = [r[0] for r in finbert_results]
    results["FinBERT"] = {
        "accuracy": accuracy_score(labels, finbert_preds),
        "f1_macro": f1_score(labels, finbert_preds, average="macro", zero_division=0),
        "total_time": finbert_time,
        "avg_time_ms": finbert_time / len(texts) * 1000,
        "report": classification_report(
            labels, finbert_preds, output_dict=True, zero_division=0
        ),
    }

    # TextBlob
    t0 = time.time()
    tb_results = analyzer.textblob.analyze_batch(texts)
    tb_time = time.time() - t0
    tb_preds = [r[0] for r in tb_results]
    results["TextBlob"] = {
        "accuracy": accuracy_score(labels, tb_preds),
        "f1_macro": f1_score(labels, tb_preds, average="macro", zero_division=0),
        "total_time": tb_time,
        "avg_time_ms": tb_time / len(texts) * 1000,
        "report": classification_report(
            labels, tb_preds, output_dict=True, zero_division=0
        ),
    }

    print(
        f"\n{'Model':<12} {'Accuracy':>10} {'F1-Macro':>10} {'Total(s)':>10} {'Per-sample(ms)':>15}"
    )
    print("-" * 60)
    for name, r in results.items():
        print(
            f"{name:<12} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
            f"{r['total_time']:>10.2f} {r['avg_time_ms']:>15.2f}"
        )

    return results


def experiment_2_hypothesis(
    analyzer: HybridSentimentAnalyzer, test_df: pd.DataFrame, n_bootstrap=1000
) -> Dict:
    """Experiment 2: Core hypothesis test (95% accuracy at 20% cost)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Core Hypothesis Test")
    print("=" * 70)

    texts = test_df["text"].tolist()
    labels = np.array(test_df["label"].tolist())

    # FinBERT baseline
    finbert_results = analyzer.finbert.analyze_batch(texts)
    finbert_preds = np.array([r[0] for r in finbert_results])
    finbert_acc = accuracy_score(labels, finbert_preds)
    finbert_f1 = f1_score(labels, finbert_preds, average="macro", zero_division=0)

    # Hybrid predictions
    hybrid_preds, use_vader, timing = analyzer.predict(texts)
    hybrid_acc = accuracy_score(labels, hybrid_preds)
    hybrid_f1 = f1_score(labels, hybrid_preds, average="macro", zero_division=0)
    finbert_usage_pct = timing["finbert_pct"]

    acc_retention = hybrid_acc / finbert_acc if finbert_acc > 0 else 0
    f1_retention = hybrid_f1 / finbert_f1 if finbert_f1 > 0 else 0

    # Bootstrap confidence intervals
    rng = np.random.RandomState(SEED)
    boot_acc_retentions = []
    boot_finbert_usages = []
    n = len(labels)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        b_labels = labels[idx]
        b_hybrid = hybrid_preds[idx]
        b_finbert = finbert_preds[idx]
        b_vader = use_vader[idx]

        b_finbert_acc = accuracy_score(b_labels, b_finbert)
        b_hybrid_acc = accuracy_score(b_labels, b_hybrid)
        if b_finbert_acc > 0:
            boot_acc_retentions.append(b_hybrid_acc / b_finbert_acc)
        boot_finbert_usages.append((~b_vader).mean() * 100)

    ci_retention = np.percentile(boot_acc_retentions, [2.5, 97.5])
    ci_usage = np.percentile(boot_finbert_usages, [2.5, 97.5])

    # McNemar's test
    hybrid_correct = hybrid_preds == labels
    finbert_correct = finbert_preds == labels
    b = np.sum(hybrid_correct & ~finbert_correct)
    c = np.sum(~hybrid_correct & finbert_correct)
    if b + c > 0:
        mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
    else:
        mcnemar_stat = 0
        mcnemar_p = 1.0

    results = {
        "finbert_accuracy": finbert_acc,
        "finbert_f1": finbert_f1,
        "hybrid_accuracy": hybrid_acc,
        "hybrid_f1": hybrid_f1,
        "accuracy_retention": acc_retention,
        "f1_retention": f1_retention,
        "finbert_usage_pct": finbert_usage_pct,
        "hypothesis_accuracy_met": acc_retention >= 0.95,
        "hypothesis_cost_met": finbert_usage_pct <= 20.0,
        "ci_retention_95": ci_retention.tolist(),
        "ci_finbert_usage_95": ci_usage.tolist(),
        "mcnemar_statistic": float(mcnemar_stat),
        "mcnemar_p_value": float(mcnemar_p),
        "timing": {k: v for k, v in timing.items() if isinstance(v, (int, float))},
    }

    print(f"\nFinBERT accuracy:    {finbert_acc:.4f} (F1: {finbert_f1:.4f})")
    print(f"Hybrid accuracy:     {hybrid_acc:.4f} (F1: {hybrid_f1:.4f})")
    print(f"Accuracy retention:  {acc_retention:.4f} (target: >=0.95)")
    print(f"  95% CI: [{ci_retention[0]:.4f}, {ci_retention[1]:.4f}]")
    print(f"FinBERT usage:       {finbert_usage_pct:.1f}% (target: <=20%)")
    print(f"  95% CI: [{ci_usage[0]:.1f}%, {ci_usage[1]:.1f}%]")
    print(f"McNemar's test:      chi2={mcnemar_stat:.2f}, p={mcnemar_p:.4f}")
    hyp = (
        "SUPPORTED"
        if results["hypothesis_accuracy_met"] and results["hypothesis_cost_met"]
        else "NOT SUPPORTED"
    )
    print(f"\nHypothesis (>=95% acc at <=20% cost): {hyp}")

    # Multi-operating-point analysis
    print("\n  --- Operating Point Sweep ---")
    print(
        f"  {'tau':>6} {'Accuracy':>10} {'Retention':>10} {'FinBERT%':>10} {'Meets Both':>12}"
    )
    print("  " + "-" * 50)

    # Get routing features for test set
    vader_features = analyzer.vader.get_features_batch(texts)
    ambiguity = analyzer.ambiguity_predictor.predict(texts)
    ling = analyzer.ling_extractor.extract_batch(texts)
    routing_scores = analyzer.gate_router.predict(vader_features, ambiguity, ling)
    vader_results_batch = analyzer.vader.analyze_batch(texts)
    vader_preds_arr = np.array([r[0] for r in vader_results_batch])

    sweep_results = []
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        use_v = routing_scores >= tau
        fb_idx = np.where(~use_v)[0]
        fb_preds = np.empty(len(texts), dtype=object)
        if len(fb_idx) > 0:
            deferred = [texts[i] for i in fb_idx]
            deferred_res = analyzer.finbert.analyze_batch(deferred)
            for idx, (lbl, _) in zip(fb_idx, deferred_res):
                fb_preds[idx] = lbl
        preds = np.where(use_v, vader_preds_arr, fb_preds)
        acc = accuracy_score(labels, preds)
        ret = acc / finbert_acc if finbert_acc > 0 else 0
        fb_pct = (~use_v).mean() * 100
        meets = ret >= 0.95 and fb_pct <= 20.0
        sweep_results.append(
            {
                "tau": tau,
                "accuracy": acc,
                "retention": ret,
                "finbert_pct": fb_pct,
                "meets_both": meets,
            }
        )
        print(
            f"  {tau:>6.2f} {acc:>10.4f} {ret:>10.4f} {fb_pct:>10.1f}% {str(meets):>12}"
        )

    results["operating_point_sweep"] = sweep_results

    return results


def experiment_3_speedup(
    analyzer: HybridSentimentAnalyzer, test_df: pd.DataFrame
) -> Dict:
    """Experiment 3: Speedup analysis."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Speedup Analysis")
    print("=" * 70)

    texts = test_df["text"].tolist()
    n = len(texts)
    results = {}

    # VADER-only
    t0 = time.time()
    analyzer.vader.analyze_batch(texts)
    vader_time = time.time() - t0
    results["VADER"] = {"total_time": vader_time, "throughput": n / vader_time}

    # FinBERT-only
    t0 = time.time()
    analyzer.finbert.analyze_batch(texts)
    finbert_time = time.time() - t0
    results["FinBERT"] = {"total_time": finbert_time, "throughput": n / finbert_time}

    # Hybrid at different thresholds
    for tau in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        t0 = time.time()
        _, use_vader, timing = analyzer.predict(texts, tau=tau)
        hybrid_time = time.time() - t0
        vader_pct = use_vader.mean() * 100
        results[f"Hybrid(tau={tau})"] = {
            "total_time": hybrid_time,
            "throughput": n / hybrid_time,
            "vader_pct": vader_pct,
        }

    print(
        f"\n{'System':<20} {'Time(s)':>10} {'Throughput':>12} {'VADER %':>10} {'Speedup':>10}"
    )
    print("-" * 65)
    for name, r in results.items():
        vp = r.get("vader_pct", "N/A")
        vp_str = f"{vp:.1f}%" if isinstance(vp, float) else vp
        speedup = finbert_time / r["total_time"] if r["total_time"] > 0 else 0
        print(
            f"{name:<20} {r['total_time']:>10.2f} {r['throughput']:>10.1f}/s {vp_str:>10} {speedup:>9.2f}x"
        )

    return results


def experiment_4_agreement_levels(
    analyzer: HybridSentimentAnalyzer, test_df: pd.DataFrame
) -> Dict:
    """Experiment 4: Performance by agreement level."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Agreement-Level Analysis")
    print("=" * 70)

    results = {}
    for tier in ["100", "75", "66", "50"]:
        subset = test_df[test_df["agreement_tier"] == tier]
        if len(subset) == 0:
            continue

        texts = subset["text"].tolist()
        labels = np.array(subset["label"].tolist())

        vader_results = analyzer.vader.analyze_batch(texts)
        vader_preds = np.array([r[0] for r in vader_results])
        vader_acc = accuracy_score(labels, vader_preds)

        finbert_results = analyzer.finbert.analyze_batch(texts)
        finbert_preds = np.array([r[0] for r in finbert_results])
        finbert_acc = accuracy_score(labels, finbert_preds)

        hybrid_preds, use_vader, _ = analyzer.predict(texts)
        hybrid_acc = accuracy_score(labels, hybrid_preds)
        vader_pct = use_vader.mean() * 100

        results[tier] = {
            "n_samples": len(subset),
            "vader_acc": vader_acc,
            "finbert_acc": finbert_acc,
            "hybrid_acc": hybrid_acc,
            "vader_usage_pct": vader_pct,
            "finbert_usage_pct": 100 - vader_pct,
        }

        print(f"\nTier {tier}% agreement (n={len(subset)}):")
        print(
            f"  VADER: {vader_acc:.4f}, FinBERT: {finbert_acc:.4f}, Hybrid: {hybrid_acc:.4f}"
        )
        print(f"  Routing: VADER={vader_pct:.1f}%, FinBERT={100 - vader_pct:.1f}%")

    return results


def experiment_5_ablation(
    analyzer: HybridSentimentAnalyzer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict:
    """Experiment 5: Ablation study of routing features."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Routing Feature Ablation")
    print("=" * 70)

    test_texts = test_df["text"].tolist()
    test_labels = np.array(test_df["label"].tolist())
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].tolist()

    # Pre-compute all features
    train_vader = analyzer.vader.analyze_batch(train_texts)
    train_vader_preds = np.array([r[0] for r in train_vader])
    train_vader_features = analyzer.vader.get_features_batch(train_texts)
    train_vader_correct = (train_vader_preds == np.array(train_labels)).astype(float)
    train_ambiguity = analyzer.ambiguity_predictor.predict(train_texts)
    train_ling = analyzer.ling_extractor.extract_batch(train_texts)

    val_vader = analyzer.vader.analyze_batch(val_texts)
    val_vader_preds = np.array([r[0] for r in val_vader])
    val_vader_features = analyzer.vader.get_features_batch(val_texts)
    val_ambiguity = analyzer.ambiguity_predictor.predict(val_texts)
    val_ling = analyzer.ling_extractor.extract_batch(val_texts)

    test_vader = analyzer.vader.analyze_batch(test_texts)
    test_vader_preds = np.array([r[0] for r in test_vader])
    test_vader_features = analyzer.vader.get_features_batch(test_texts)
    test_ambiguity = analyzer.ambiguity_predictor.predict(test_texts)
    test_ling = analyzer.ling_extractor.extract_batch(test_texts)

    val_finbert = analyzer.finbert.analyze_batch(val_texts)
    val_finbert_preds = np.array([r[0] for r in val_finbert])
    test_finbert = analyzer.finbert.analyze_batch(test_texts)
    test_finbert_preds = np.array([r[0] for r in test_finbert])

    ablation_configs = {
        "Full model": {"vader": True, "ambiguity": True, "linguistic": True},
        "No VADER features": {"vader": False, "ambiguity": True, "linguistic": True},
        "No ambiguity": {"vader": True, "ambiguity": False, "linguistic": True},
        "No linguistic": {"vader": True, "ambiguity": True, "linguistic": False},
        "VADER only": {"vader": True, "ambiguity": False, "linguistic": False},
    }

    results = {}
    for name, config in ablation_configs.items():

        def apply_mask(vf, af, lf, cfg):
            return (
                vf if cfg["vader"] else np.zeros_like(vf),
                af if cfg["ambiguity"] else np.zeros_like(af),
                lf if cfg["linguistic"] else np.zeros_like(lf),
            )

        tr_vf, tr_af, tr_lf = apply_mask(
            train_vader_features, train_ambiguity, train_ling, config
        )
        gate = GateMLPRouter(input_dim=15, epochs=100)
        gate.train(tr_vf, tr_af, tr_lf, train_vader_correct)

        va_vf, va_af, va_lf = apply_mask(
            val_vader_features, val_ambiguity, val_ling, config
        )
        val_scores = gate.predict(va_vf, va_af, va_lf)

        # Find best tau for this ablation
        val_finbert_acc = accuracy_score(np.array(val_labels), val_finbert_preds)
        opt = CostAwareOptimizer()
        target = opt.find_target_operating_point(
            val_scores,
            val_vader_preds,
            val_finbert_preds,
            np.array(val_labels),
            val_finbert_acc,
        )
        tau = target["tau"] if target else 0.5

        te_vf, te_af, te_lf = apply_mask(
            test_vader_features, test_ambiguity, test_ling, config
        )
        test_scores = gate.predict(te_vf, te_af, te_lf)
        use_vader = test_scores >= tau
        preds = np.where(use_vader, test_vader_preds, test_finbert_preds)

        acc = accuracy_score(test_labels, preds)
        f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
        finbert_pct = (~use_vader).mean() * 100

        results[name] = {
            "accuracy": acc,
            "f1_macro": f1,
            "finbert_pct": finbert_pct,
            "tau": tau,
        }
        print(f"{name:<25} Acc={acc:.4f}  F1={f1:.4f}  FinBERT={finbert_pct:.1f}%")

    return results


def experiment_6_pareto(
    analyzer: HybridSentimentAnalyzer, test_df: pd.DataFrame
) -> Dict:
    """Experiment 6: Pareto frontier - accuracy vs cost."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Pareto Frontier")
    print("=" * 70)

    texts = test_df["text"].tolist()
    labels = np.array(test_df["label"].tolist())

    vader_results = analyzer.vader.analyze_batch(texts)
    vader_preds = np.array([r[0] for r in vader_results])
    vader_features = analyzer.vader.get_features_batch(texts)
    ambiguity = analyzer.ambiguity_predictor.predict(texts)
    ling = analyzer.ling_extractor.extract_batch(texts)

    finbert_results = analyzer.finbert.analyze_batch(texts)
    finbert_preds = np.array([r[0] for r in finbert_results])

    # Finance-aware routing scores
    routing_scores = analyzer.gate_router.predict(vader_features, ambiguity, ling)
    pareto = analyzer.cost_optimizer.get_pareto_frontier(
        routing_scores, vader_preds, finbert_preds, labels
    )

    # Naive confidence routing comparison
    vader_confs = np.array([r[1] for r in vader_results])
    naive_pareto = []
    for tau in np.arange(0.0, 1.01, 0.01):
        use_vader = vader_confs >= tau
        preds = np.where(use_vader, vader_preds, finbert_preds)
        acc = accuracy_score(labels, preds)
        finbert_pct = (~use_vader).mean() * 100
        naive_pareto.append(
            {"tau": float(tau), "accuracy": acc, "finbert_pct": finbert_pct}
        )

    results = {"finance_aware": pareto, "naive": naive_pareto}
    print(f"Finance-aware: best acc={max(p['accuracy'] for p in pareto):.4f}")
    print(f"Naive: best acc={max(p['accuracy'] for p in naive_pareto):.4f}")

    return results


def experiment_7_distribution_shift(
    analyzer: HybridSentimentAnalyzer, full_df: pd.DataFrame
) -> Dict:
    """Experiment 7: Distribution shift - train on easy, test on hard."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Distribution Shift Analysis")
    print("=" * 70)

    easy_df = full_df[full_df["agreement_tier"] == "100"]
    hard_df = full_df[full_df["agreement_tier"] == "50"]

    if len(hard_df) == 0:
        print("No hard samples (50-only) found.")
        return {}

    results = {}
    for name, subset_df in [("Easy (100%)", easy_df), ("Hard (50% only)", hard_df)]:
        texts = subset_df["text"].tolist()
        labels = np.array(subset_df["label"].tolist())

        vader_results = analyzer.vader.analyze_batch(texts)
        vader_preds = np.array([r[0] for r in vader_results])

        finbert_results = analyzer.finbert.analyze_batch(texts)
        finbert_preds = np.array([r[0] for r in finbert_results])

        hybrid_preds, use_vader, _ = analyzer.predict(texts)

        results[name] = {
            "n_samples": len(texts),
            "vader_acc": accuracy_score(labels, vader_preds),
            "finbert_acc": accuracy_score(labels, finbert_preds),
            "hybrid_acc": accuracy_score(labels, hybrid_preds),
            "vader_f1": f1_score(labels, vader_preds, average="macro", zero_division=0),
            "finbert_f1": f1_score(
                labels, finbert_preds, average="macro", zero_division=0
            ),
            "hybrid_f1": f1_score(
                labels, hybrid_preds, average="macro", zero_division=0
            ),
            "vader_usage_pct": use_vader.mean() * 100,
        }

        print(f"\n{name} (n={len(texts)}):")
        print(
            f"  VADER: acc={results[name]['vader_acc']:.4f}, F1={results[name]['vader_f1']:.4f}"
        )
        print(
            f"  FinBERT: acc={results[name]['finbert_acc']:.4f}, F1={results[name]['finbert_f1']:.4f}"
        )
        print(
            f"  Hybrid: acc={results[name]['hybrid_acc']:.4f}, F1={results[name]['hybrid_f1']:.4f}"
        )
        print(f"  VADER usage: {results[name]['vader_usage_pct']:.1f}%")

    return results


def experiment_8_per_class(
    analyzer: HybridSentimentAnalyzer, test_df: pd.DataFrame
) -> Dict:
    """Experiment 8: Per-class breakdown and confusion matrices."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Per-Class Breakdown")
    print("=" * 70)

    texts = test_df["text"].tolist()
    labels = np.array(test_df["label"].tolist())

    vader_results = analyzer.vader.analyze_batch(texts)
    vader_preds = np.array([r[0] for r in vader_results])

    finbert_results = analyzer.finbert.analyze_batch(texts)
    finbert_preds = np.array([r[0] for r in finbert_results])

    hybrid_preds, _, _ = analyzer.predict(texts)

    results = {}
    for name, preds in [
        ("VADER", vader_preds),
        ("FinBERT", finbert_preds),
        ("Hybrid", hybrid_preds),
    ]:
        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(labels, preds, labels=LABEL_LIST)
        results[name] = {"report": report, "confusion_matrix": cm.tolist()}
        print(f"\n{name}:")
        print(classification_report(labels, preds, zero_division=0))

    return results


# ============================================================
# Visualization
# ============================================================


def plot_baseline_comparison(exp1_results: Dict, save_dir: Path):
    """Plot 1: Baseline comparison bar chart."""
    models = list(exp1_results.keys())
    accs = [exp1_results[m]["accuracy"] for m in models]
    f1s = [exp1_results[m]["f1_macro"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(models))
    width = 0.35

    axes[0].bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
    axes[0].bar(x + width / 2, f1s, width, label="F1-Macro", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Model Performance Comparison")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        axes[0].text(i - width / 2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
        axes[0].text(i + width / 2, f + 0.02, f"{f:.3f}", ha="center", fontsize=9)

    times = [exp1_results[m]["avg_time_ms"] for m in models]
    bars = axes[1].bar(models, times, color=["steelblue", "coral", "seagreen"])
    axes[1].set_ylabel("Avg. Time per Sample (ms)")
    axes[1].set_title("Inference Latency")
    for bar, t in zip(bars, times):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{t:.2f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_dir / "01_baseline_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_pareto_frontier(exp6_results: Dict, save_dir: Path):
    """Plot 2: Pareto frontier - accuracy vs cost."""
    fig, ax = plt.subplots(figsize=(8, 6))

    fa = exp6_results["finance_aware"]
    fa_x = [p["finbert_pct"] for p in fa]
    fa_y = [p["accuracy"] for p in fa]
    ax.plot(
        fa_x,
        fa_y,
        "o-",
        color="steelblue",
        markersize=2,
        label="Finance-Aware Router",
        alpha=0.7,
    )

    naive = exp6_results["naive"]
    na_x = [p["finbert_pct"] for p in naive]
    na_y = [p["accuracy"] for p in naive]
    ax.plot(
        na_x,
        na_y,
        "s-",
        color="coral",
        markersize=2,
        label="Naive Confidence Router",
        alpha=0.7,
    )

    ax.axvline(x=20, color="gray", linestyle="--", alpha=0.5, label="20% Cost Target")
    ax.set_xlabel("FinBERT Usage (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto Frontier: Accuracy vs. Computational Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "02_pareto_frontier.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_threshold_sensitivity(exp6_results: Dict, save_dir: Path):
    """Plot 3: Threshold sensitivity curve."""
    fa = exp6_results["finance_aware"]
    taus = [p["tau"] for p in fa]
    accs = [p["accuracy"] for p in fa]
    f1s = [p["f1_macro"] for p in fa]
    finbert_pcts = [p["finbert_pct"] for p in fa]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(taus, accs, "b-", label="Accuracy", linewidth=2)
    ax1.plot(taus, f1s, "b--", label="F1-Macro", linewidth=2, alpha=0.7)
    ax2.plot(taus, finbert_pcts, "r-", label="FinBERT Usage %", linewidth=2)

    ax1.set_xlabel("Threshold (tau)")
    ax1.set_ylabel("Performance", color="blue")
    ax2.set_ylabel("FinBERT Usage (%)", color="red")
    ax1.set_title("Threshold Sensitivity Analysis")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "03_threshold_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_agreement_routing(exp4_results: Dict, save_dir: Path):
    """Plot 4: Agreement-level routing behavior."""
    tiers = sorted(exp4_results.keys(), key=lambda x: int(x))
    vader_pcts = [exp4_results[t]["vader_usage_pct"] for t in tiers]
    finbert_pcts = [exp4_results[t]["finbert_usage_pct"] for t in tiers]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(tiers))
    tier_labels = [f"{t}% Agree" for t in tiers]
    axes[0].bar(x, vader_pcts, label="VADER", color="steelblue")
    axes[0].bar(x, finbert_pcts, bottom=vader_pcts, label="FinBERT", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tier_labels)
    axes[0].set_ylabel("Routing Distribution (%)")
    axes[0].set_title("Model Routing by Agreement Level")
    axes[0].legend()

    vader_accs = [exp4_results[t]["vader_acc"] for t in tiers]
    finbert_accs = [exp4_results[t]["finbert_acc"] for t in tiers]
    hybrid_accs = [exp4_results[t]["hybrid_acc"] for t in tiers]
    width = 0.25
    axes[1].bar(x - width, vader_accs, width, label="VADER", color="steelblue")
    axes[1].bar(x, finbert_accs, width, label="FinBERT", color="coral")
    axes[1].bar(x + width, hybrid_accs, width, label="Hybrid", color="seagreen")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tier_labels)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy by Agreement Level")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "04_agreement_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ablation(exp5_results: Dict, save_dir: Path):
    """Plot 5: Ablation study bar chart."""
    configs = list(exp5_results.keys())
    accs = [exp5_results[c]["accuracy"] for c in configs]
    f1s = [exp5_results[c]["f1_macro"] for c in configs]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(configs))
    width = 0.35

    ax.bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
    ax.bar(x + width / 2, f1s, width, label="F1-Macro", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Routing Feature Ablation Study")
    ax.legend()
    ax.set_ylim(min(min(accs), min(f1s)) - 0.05, 1.05)

    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / "05_ablation_study.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_speedup(exp3_results: Dict, save_dir: Path):
    """Plot 6: Speedup comparison."""
    systems = list(exp3_results.keys())
    throughputs = [exp3_results[s]["throughput"] for s in systems]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue"] + ["coral"] + ["seagreen"] * (len(systems) - 2)
    bars = ax.bar(range(len(systems)), throughputs, color=colors)
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems, rotation=30, ha="right")
    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title("Inference Throughput Comparison")

    for bar, t in zip(bars, throughputs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{t:.1f}",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_dir / "06_speedup_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(exp8_results: Dict, save_dir: Path):
    """Plot 7: Confusion matrices for all three systems."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, data) in zip(axes, exp8_results.items()):
        cm = np.array(data["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABEL_LIST,
            yticklabels=LABEL_LIST,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name} Confusion Matrix")

    plt.tight_layout()
    plt.savefig(save_dir / "07_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_calibration(
    analyzer: HybridSentimentAnalyzer, test_df: pd.DataFrame, save_dir: Path
):
    """Plot 8: Calibration reliability diagram."""
    texts = test_df["text"].tolist()
    labels = np.array([LABEL2ID[l] for l in test_df["label"].tolist()])

    logits, uncal_probs = analyzer.finbert.get_logits_batch(texts)
    cal_probs = analyzer.temp_scaler.calibrate(logits)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, probs, title in [
        (axes[0], uncal_probs, "Before Calibration"),
        (
            axes[1],
            cal_probs,
            f"After Temperature Scaling (T={analyzer.temp_scaler.temperature:.2f})",
        ),
    ]:
        pred_classes = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)
        correct = (pred_classes == labels).astype(float)

        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_accs = []
        bin_confs = []
        for i in range(n_bins):
            mask = (max_probs >= bins[i]) & (max_probs < bins[i + 1])
            if mask.sum() > 0:
                bin_accs.append(correct[mask].mean())
                bin_confs.append(max_probs[mask].mean())
            else:
                bin_accs.append(0)
                bin_confs.append((bins[i] + bins[i + 1]) / 2)

        ax.bar(
            bin_confs,
            bin_accs,
            width=0.08,
            alpha=0.6,
            color="steelblue",
            label="Accuracy",
        )
        ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "08_calibration_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_distribution_shift(exp7_results: Dict, save_dir: Path):
    """Plot 9: Distribution shift comparison."""
    if not exp7_results:
        return

    subsets = list(exp7_results.keys())
    models = ["VADER", "FinBERT", "Hybrid"]
    colors = ["steelblue", "coral", "seagreen"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(subsets))
    width = 0.25
    for i, model in enumerate(models):
        key = f"{model.lower()}_acc"
        accs = [exp7_results[s][key] for s in subsets]
        axes[0].bar(x + i * width, accs, width, label=model, color=colors[i])
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(subsets, fontsize=9)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy Under Distribution Shift")
    axes[0].legend()

    for i, model in enumerate(models):
        key = f"{model.lower()}_f1"
        f1s = [exp7_results[s][key] for s in subsets]
        axes[1].bar(x + i * width, f1s, width, label=model, color=colors[i])
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(subsets, fontsize=9)
    axes[1].set_ylabel("F1-Macro")
    axes[1].set_title("F1-Macro Under Distribution Shift")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "09_distribution_shift.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid FinBERT-VADER Sentiment Analysis"
    )
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--gate-epochs", type=int, default=100)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    set_seed()

    print(f"Device: {DEVICE}")
    print(f"Results directory: {results_dir}")

    # ---- Data ----
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    df = load_all_phrasebank(args.data_dir)
    train_df, val_df, test_df = split_data(
        df, test_size=args.test_split, val_size=args.val_split
    )

    # ---- Initialize and Train ----
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE")
    print("=" * 70)
    analyzer = HybridSentimentAnalyzer()
    analyzer.gate_router = GateMLPRouter(
        input_dim=15, lr=args.gate_lr, epochs=args.gate_epochs
    )
    opt_results = analyzer.train_pipeline(train_df, val_df)

    # ---- Run All Experiments ----
    all_results = {}

    all_results["exp1_baselines"] = experiment_1_baselines(analyzer, test_df)
    all_results["exp2_hypothesis"] = experiment_2_hypothesis(
        analyzer, test_df, n_bootstrap=args.n_bootstrap
    )
    all_results["exp3_speedup"] = experiment_3_speedup(analyzer, test_df)
    all_results["exp4_agreement"] = experiment_4_agreement_levels(analyzer, test_df)
    all_results["exp5_ablation"] = experiment_5_ablation(
        analyzer, train_df, val_df, test_df
    )
    all_results["exp6_pareto"] = experiment_6_pareto(analyzer, test_df)
    all_results["exp7_shift"] = experiment_7_distribution_shift(analyzer, df)
    all_results["exp8_per_class"] = experiment_8_per_class(analyzer, test_df)
    all_results["cost_aware_optimization"] = {str(k): v for k, v in opt_results.items()}

    # ---- Plots ----
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_baseline_comparison(all_results["exp1_baselines"], results_dir)
    plot_pareto_frontier(all_results["exp6_pareto"], results_dir)
    plot_threshold_sensitivity(all_results["exp6_pareto"], results_dir)
    plot_agreement_routing(all_results["exp4_agreement"], results_dir)
    plot_ablation(all_results["exp5_ablation"], results_dir)
    plot_speedup(all_results["exp3_speedup"], results_dir)
    plot_confusion_matrices(all_results["exp8_per_class"], results_dir)
    plot_calibration(analyzer, test_df, results_dir)
    plot_distribution_shift(all_results["exp7_shift"], results_dir)
    print(f"All plots saved to {results_dir}/")

    # ---- Save Results ----
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    results_path = results_dir / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"Results saved to {results_path}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    exp2 = all_results["exp2_hypothesis"]
    print(f"FinBERT Accuracy:   {exp2['finbert_accuracy']:.4f}")
    print(f"Hybrid Accuracy:    {exp2['hybrid_accuracy']:.4f}")
    print(f"Accuracy Retention: {exp2['accuracy_retention']:.4f} (target: >=0.95)")
    print(f"FinBERT Usage:      {exp2['finbert_usage_pct']:.1f}% (target: <=20%)")
    hyp = (
        "SUPPORTED"
        if exp2["hypothesis_accuracy_met"] and exp2["hypothesis_cost_met"]
        else "NOT SUPPORTED"
    )
    print(f"Hypothesis:         {hyp}")

    return all_results


if __name__ == "__main__":
    main()
