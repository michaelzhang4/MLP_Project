"""
Hybrid FinBERT-VADER Sentiment Analysis with SetFit Confidence Router

This script combines the speed of VADER with the accuracy of FinBERT by using
a SetFit model to learn when VADER's predictions are reliable enough to skip
the expensive FinBERT inference.

Architecture:
1. VADER provides fast initial sentiment + confidence scores
2. SetFit classifier learns to predict when VADER is accurate enough
3. Only uncertain cases are routed to FinBERT for final prediction

This achieves significant speedup while maintaining high accuracy.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class SentimentResult:
    """Container for sentiment prediction results."""

    text: str
    predicted_label: str
    confidence: float
    model_used: str
    inference_time: float


class VADERAnalyzer:
    """VADER sentiment analyzer with confidence scoring."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.label_map = {-1: "negative", 0: "neutral", 1: "positive"}

    def analyze(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Analyze sentiment using VADER.

        Returns:
            Tuple of (predicted_label, confidence, raw_scores)
        """
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        # Determine label based on compound score
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        # Calculate confidence as the absolute compound score
        # Higher absolute values indicate more confident predictions
        confidence = abs(compound)

        return label, confidence, scores

    def get_features(self, text: str) -> np.ndarray:
        """Extract features for SetFit training."""
        scores = self.analyzer.polarity_scores(text)
        return np.array(
            [
                scores["pos"],
                scores["neg"],
                scores["neu"],
                scores["compound"],
                abs(scores["compound"]),  # confidence
                len(text.split()),  # word count
            ]
        )


class FinBERTAnalyzer:
    """FinBERT sentiment analyzer."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.label_map = {0: "positive", 1: "negative", 2: "neutral"}

    def analyze(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment using FinBERT.

        Returns:
            Tuple of (predicted_label, confidence)
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

        label = self.label_map[predicted.item()]
        return label, confidence.item()

    def analyze_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Tuple[str, float]]:
        """Batch inference for efficiency."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
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


class ConfidenceRouter:
    """
    SetFit-based router that learns when to trust VADER predictions.

    The router is trained on examples where we know the ground truth,
    learning to predict whether VADER will be correct for a given input.
    """

    def __init__(
        self, model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    ):
        self.model_name = model_name
        self.model: Optional[SetFitModel] = None
        self.threshold = 0.5

    def prepare_training_data(
        self, texts: List[str], true_labels: List[str], vader_predictions: List[str]
    ) -> Dataset:
        """
        Prepare training data for the router.
        Label 1 = VADER is correct, use VADER
        Label 0 = VADER is wrong, use FinBERT
        """
        labels = [
            1 if pred == true else 0
            for pred, true in zip(vader_predictions, true_labels)
        ]
        return Dataset.from_dict({"text": texts, "label": labels})

    def train(
        self,
        train_texts: List[str],
        train_true_labels: List[str],
        train_vader_preds: List[str],
        eval_texts: Optional[List[str]] = None,
        eval_true_labels: Optional[List[str]] = None,
        eval_vader_preds: Optional[List[str]] = None,
        num_iterations: int = 20,
        num_epochs: int = 1,
    ):
        """Train the SetFit router model."""
        train_dataset = self.prepare_training_data(
            train_texts, train_true_labels, train_vader_preds
        )

        eval_dataset = None
        if eval_texts is not None:
            eval_dataset = self.prepare_training_data(
                eval_texts, eval_true_labels, eval_vader_preds
            )

        self.model = SetFitModel.from_pretrained(self.model_name)

        args = TrainingArguments(
            batch_size=16,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="no",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()

    def should_use_vader(self, text: str) -> Tuple[bool, float]:
        """
        Predict whether VADER will be accurate for this text.

        Returns:
            Tuple of (use_vader, confidence)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        probs = self.model.predict_proba([text])[0]
        vader_correct_prob = probs[1] if len(probs) > 1 else probs[0]

        return vader_correct_prob >= self.threshold, vader_correct_prob

    def should_use_vader_batch(self, texts: List[str]) -> List[Tuple[bool, float]]:
        """Batch prediction for routing decisions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        probs = self.model.predict_proba(texts)
        results = []
        for prob in probs:
            vader_correct_prob = prob[1] if len(prob) > 1 else prob[0]
            results.append((vader_correct_prob >= self.threshold, vader_correct_prob))

        return results

    def set_threshold(self, threshold: float):
        """Adjust the routing threshold."""
        self.threshold = threshold

    def save(self, path: str):
        """Save the trained model."""
        if self.model is not None:
            self.model.save_pretrained(path)

    def load(self, path: str):
        """Load a trained model."""
        self.model = SetFitModel.from_pretrained(path)


class HybridSentimentAnalyzer:
    """
    Hybrid sentiment analyzer combining VADER speed with FinBERT accuracy.

    Uses a SetFit-trained router to decide when VADER is sufficient,
    falling back to FinBERT only when necessary.
    """

    def __init__(
        self,
        router_model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
        finbert_model_name: str = "ProsusAI/finbert",
        confidence_threshold: float = 0.5,
    ):
        self.vader = VADERAnalyzer()
        self.finbert = FinBERTAnalyzer(finbert_model_name)
        self.router = ConfidenceRouter(router_model_name)
        self.router.set_threshold(confidence_threshold)

        # Statistics tracking
        self.stats = {
            "vader_used": 0,
            "finbert_used": 0,
            "total_vader_time": 0.0,
            "total_finbert_time": 0.0,
            "total_router_time": 0.0,
        }

    def train_router(
        self, texts: List[str], true_labels: List[str], val_split: float = 0.1, **kwargs
    ):
        """
        Train the routing model on labeled data.

        Args:
            texts: Training texts
            true_labels: Ground truth labels
            val_split: Validation split ratio
            **kwargs: Additional arguments for router training
        """
        # Get VADER predictions for all training data
        vader_preds = []
        for text in texts:
            label, _, _ = self.vader.analyze(text)
            vader_preds.append(label)

        # Split data
        if val_split > 0:
            train_texts, val_texts, train_labels, val_labels, train_vader, val_vader = (
                train_test_split(
                    texts,
                    true_labels,
                    vader_preds,
                    test_size=val_split,
                    random_state=42,
                )
            )
        else:
            train_texts, train_labels, train_vader = texts, true_labels, vader_preds
            val_texts, val_labels, val_vader = None, None, None

        # Train the router
        self.router.train(
            train_texts,
            train_labels,
            train_vader,
            val_texts,
            val_labels,
            val_vader,
            **kwargs,
        )

        # Print VADER baseline accuracy
        correct = sum(1 for p, t in zip(vader_preds, true_labels) if p == t)
        print(f"VADER baseline accuracy: {correct / len(true_labels):.4f}")

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment using the hybrid approach.

        Returns:
            SentimentResult with prediction details
        """
        start_time = time.time()

        # First, get VADER prediction
        vader_start = time.time()
        vader_label, vader_conf, _ = self.vader.analyze(text)
        vader_time = time.time() - vader_start
        self.stats["total_vader_time"] += vader_time

        # Use router to decide
        router_start = time.time()
        use_vader, router_conf = self.router.should_use_vader(text)
        router_time = time.time() - router_start
        self.stats["total_router_time"] += router_time

        if use_vader:
            self.stats["vader_used"] += 1
            return SentimentResult(
                text=text,
                predicted_label=vader_label,
                confidence=vader_conf,
                model_used="vader",
                inference_time=time.time() - start_time,
            )
        else:
            # Fall back to FinBERT
            finbert_start = time.time()
            finbert_label, finbert_conf = self.finbert.analyze(text)
            finbert_time = time.time() - finbert_start
            self.stats["total_finbert_time"] += finbert_time
            self.stats["finbert_used"] += 1

            return SentimentResult(
                text=text,
                predicted_label=finbert_label,
                confidence=finbert_conf,
                model_used="finbert",
                inference_time=time.time() - start_time,
            )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Batch analysis with optimized routing."""
        start_time = time.time()
        results = [None] * len(texts)

        # Get all VADER predictions
        vader_start = time.time()
        vader_results = []
        for text in texts:
            label, conf, _ = self.vader.analyze(text)
            vader_results.append((label, conf))
        vader_time = time.time() - vader_start
        self.stats["total_vader_time"] += vader_time

        # Route all texts
        router_start = time.time()
        routing_decisions = self.router.should_use_vader_batch(texts)
        router_time = time.time() - router_start
        self.stats["total_router_time"] += router_time

        # Collect texts that need FinBERT
        finbert_indices = []
        finbert_texts = []

        for i, (use_vader, router_conf) in enumerate(routing_decisions):
            if use_vader:
                self.stats["vader_used"] += 1
                results[i] = SentimentResult(
                    text=texts[i],
                    predicted_label=vader_results[i][0],
                    confidence=vader_results[i][1],
                    model_used="vader",
                    inference_time=0,  # Will update at end
                )
            else:
                finbert_indices.append(i)
                finbert_texts.append(texts[i])

        # Batch process FinBERT texts
        if finbert_texts:
            finbert_start = time.time()
            finbert_results = self.finbert.analyze_batch(finbert_texts)
            finbert_time = time.time() - finbert_start
            self.stats["total_finbert_time"] += finbert_time
            self.stats["finbert_used"] += len(finbert_texts)

            for idx, (label, conf) in zip(finbert_indices, finbert_results):
                results[idx] = SentimentResult(
                    text=texts[idx],
                    predicted_label=label,
                    confidence=conf,
                    model_used="finbert",
                    inference_time=0,
                )

        total_time = time.time() - start_time
        avg_time = total_time / len(texts)
        for result in results:
            result.inference_time = avg_time

        return results

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        total = self.stats["vader_used"] + self.stats["finbert_used"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "vader_percentage": self.stats["vader_used"] / total * 100,
            "finbert_percentage": self.stats["finbert_used"] / total * 100,
            "avg_vader_time": self.stats["total_vader_time"]
            / max(1, self.stats["vader_used"]),
            "avg_finbert_time": self.stats["total_finbert_time"]
            / max(1, self.stats["finbert_used"]),
        }

    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            "vader_used": 0,
            "finbert_used": 0,
            "total_vader_time": 0.0,
            "total_finbert_time": 0.0,
            "total_router_time": 0.0,
        }

    def set_confidence_threshold(self, threshold: float):
        """Adjust the routing threshold (higher = more VADER usage)."""
        self.router.set_threshold(threshold)

    def save_router(self, path: str):
        """Save the trained router model."""
        self.router.save(path)

    def load_router(self, path: str):
        """Load a pre-trained router model."""
        self.router.load(path)


def load_financial_phrasebank(
    filepath: str, encoding: str = "latin-1"
) -> Tuple[List[str], List[str]]:
    """
    Load the FinancialPhraseBank dataset.

    Args:
        filepath: Path to the dataset file
        encoding: File encoding (default latin-1 for special characters)

    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []

    with open(filepath, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if "@" in line:
                parts = line.rsplit("@", 1)
                if len(parts) == 2:
                    text, label = parts
                    texts.append(text.strip())
                    labels.append(label.strip())

    return texts, labels


def evaluate_models(
    texts: List[str], true_labels: List[str], hybrid_analyzer: HybridSentimentAnalyzer
) -> Dict:
    """
    Evaluate and compare model performance.

    Returns performance metrics for VADER-only, FinBERT-only, and Hybrid approaches.
    """
    results = {}

    # VADER-only evaluation
    print("Evaluating VADER-only...")
    vader_start = time.time()
    vader_preds = []
    for text in texts:
        label, _, _ = hybrid_analyzer.vader.analyze(text)
        vader_preds.append(label)
    vader_time = time.time() - vader_start

    results["vader"] = {
        "accuracy": accuracy_score(true_labels, vader_preds),
        "f1_macro": f1_score(true_labels, vader_preds, average="macro"),
        "total_time": vader_time,
        "avg_time_per_sample": vader_time / len(texts),
        "report": classification_report(true_labels, vader_preds, output_dict=True),
    }

    # FinBERT-only evaluation
    print("Evaluating FinBERT-only...")
    finbert_start = time.time()
    finbert_results = hybrid_analyzer.finbert.analyze_batch(texts)
    finbert_preds = [r[0] for r in finbert_results]
    finbert_time = time.time() - finbert_start

    results["finbert"] = {
        "accuracy": accuracy_score(true_labels, finbert_preds),
        "f1_macro": f1_score(true_labels, finbert_preds, average="macro"),
        "total_time": finbert_time,
        "avg_time_per_sample": finbert_time / len(texts),
        "report": classification_report(true_labels, finbert_preds, output_dict=True),
    }

    # Hybrid evaluation
    print("Evaluating Hybrid model...")
    hybrid_analyzer.reset_stats()
    hybrid_start = time.time()
    hybrid_results = hybrid_analyzer.analyze_batch(texts)
    hybrid_preds = [r.predicted_label for r in hybrid_results]
    hybrid_time = time.time() - hybrid_start

    results["hybrid"] = {
        "accuracy": accuracy_score(true_labels, hybrid_preds),
        "f1_macro": f1_score(true_labels, hybrid_preds, average="macro"),
        "total_time": hybrid_time,
        "avg_time_per_sample": hybrid_time / len(texts),
        "stats": hybrid_analyzer.get_stats(),
        "report": classification_report(true_labels, hybrid_preds, output_dict=True),
    }

    # Calculate speedup
    results["speedup_vs_finbert"] = (
        finbert_time / hybrid_time if hybrid_time > 0 else float("inf")
    )

    return results


def print_results(results: Dict):
    """Print formatted evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- VADER Only ---")
    print(f"Accuracy: {results['vader']['accuracy']:.4f}")
    print(f"F1 (macro): {results['vader']['f1_macro']:.4f}")
    print(f"Total time: {results['vader']['total_time']:.2f}s")
    print(f"Avg time/sample: {results['vader']['avg_time_per_sample'] * 1000:.2f}ms")

    print("\n--- FinBERT Only ---")
    print(f"Accuracy: {results['finbert']['accuracy']:.4f}")
    print(f"F1 (macro): {results['finbert']['f1_macro']:.4f}")
    print(f"Total time: {results['finbert']['total_time']:.2f}s")
    print(f"Avg time/sample: {results['finbert']['avg_time_per_sample'] * 1000:.2f}ms")

    print("\n--- Hybrid (VADER + FinBERT + SetFit Router) ---")
    print(f"Accuracy: {results['hybrid']['accuracy']:.4f}")
    print(f"F1 (macro): {results['hybrid']['f1_macro']:.4f}")
    print(f"Total time: {results['hybrid']['total_time']:.2f}s")
    print(f"Avg time/sample: {results['hybrid']['avg_time_per_sample'] * 1000:.2f}ms")

    stats = results["hybrid"]["stats"]
    print(f"\nRouting statistics:")
    print(f"  - VADER used: {stats['vader_used']} ({stats['vader_percentage']:.1f}%)")
    print(
        f"  - FinBERT used: {stats['finbert_used']} ({stats['finbert_percentage']:.1f}%)"
    )

    print(f"\n>>> Speedup vs FinBERT-only: {results['speedup_vs_finbert']:.2f}x")
    print("=" * 60)


def main():
    """Main function to run the hybrid sentiment analysis experiment."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid FinBERT-VADER Sentiment Analysis"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
        help="Path to FinancialPhraseBank data file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Router confidence threshold (higher = more VADER usage)",
    )
    parser.add_argument(
        "--test-split", type=float, default=0.2, help="Test set split ratio"
    )
    parser.add_argument(
        "--router-iterations",
        type=int,
        default=20,
        help="Number of SetFit training iterations",
    )
    parser.add_argument(
        "--save-router",
        type=str,
        default=None,
        help="Path to save trained router model",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_file}...")
    texts, labels = load_financial_phrasebank(args.data_file)
    print(f"Loaded {len(texts)} samples")

    # Print label distribution
    from collections import Counter

    label_counts = Counter(labels)
    print(f"Label distribution: {dict(label_counts)}")

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=args.test_split, random_state=42, stratify=labels
    )
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

    # Initialize hybrid analyzer
    print("\nInitializing models...")
    analyzer = HybridSentimentAnalyzer(confidence_threshold=args.confidence_threshold)

    # Train the router
    print("\nTraining SetFit router...")
    analyzer.train_router(
        train_texts, train_labels, num_iterations=args.router_iterations
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_models(test_texts, test_labels, analyzer)

    # Print results
    print_results(results)

    # Save router if requested
    if args.save_router:
        print(f"\nSaving router model to {args.save_router}...")
        analyzer.save_router(args.save_router)

    # Threshold sensitivity analysis
    print("\n" + "=" * 60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 60)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for threshold in thresholds:
        analyzer.set_confidence_threshold(threshold)
        analyzer.reset_stats()

        hybrid_results = analyzer.analyze_batch(test_texts)
        hybrid_preds = [r.predicted_label for r in hybrid_results]

        acc = accuracy_score(test_labels, hybrid_preds)
        stats = analyzer.get_stats()

        print(
            f"Threshold {threshold:.1f}: Accuracy={acc:.4f}, "
            f"VADER={stats['vader_percentage']:.1f}%, "
            f"FinBERT={stats['finbert_percentage']:.1f}%"
        )

    return results


if __name__ == "__main__":
    main()
