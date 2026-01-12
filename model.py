"""
Naive Bayes Classifier Module
Trains and evaluates email classification model
"""
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class EmailClassifier:
    """Naive Bayes classifier for email relevancy"""

    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize classifier

        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to extract (default: unigrams and bigrams)
        """
        # TF-IDF vectorizer for text features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
        )

        # Multinomial Naive Bayes classifier
        self.classifier = MultinomialNB(alpha=0.1)  # Laplace smoothing

        self.is_trained = False
        self.label_map = {'not_relevant': 0, 'relevant': 1}
        self.reverse_label_map = {0: 'not_relevant', 1: 'relevant'}

    def prepare_data(self, emails: List[Dict], test_size: float = 0.2, random_state: int = 42):
        """
        Prepare and split data for training

        Args:
            emails: List of email dictionaries
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Extract text and labels
        texts = [email['text'] for email in emails]
        labels = [self.label_map[email['label']] for email in emails]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels  # Maintain class distribution
        )

        return X_train, X_test, y_train, y_test

    def train(self, X_train: List[str], y_train: List[int]):
        """
        Train the classifier

        Args:
            X_train: Training texts
            y_train: Training labels
        """
        print("Vectorizing training data...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)

        print(f"Training Naive Bayes classifier...")
        print(f"  Features: {X_train_vectorized.shape[1]}")
        print(f"  Training samples: {X_train_vectorized.shape[0]}")

        self.classifier.fit(X_train_vectorized, y_train)
        self.is_trained = True

        print("Training complete!")

    def evaluate(self, X_test: List[str], y_test: List[int]) -> Dict:
        """
        Evaluate the classifier

        Args:
            X_test: Test texts
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        print("\nEvaluating model...")
        X_test_vectorized = self.vectorizer.transform(X_test)
        y_pred = self.classifier.predict(X_test_vectorized)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['not_relevant', 'relevant']
        ))

        print("\nConfusion Matrix:")
        print("                Predicted")
        print("                Not Rel  Relevant")
        print(f"Actual Not Rel  {conf_matrix[0][0]:7d}  {conf_matrix[0][1]:8d}")
        print(f"Actual Relevant {conf_matrix[1][0]:7d}  {conf_matrix[1][1]:8d}")

        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist()
        }

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict relevancy for new emails

        Args:
            texts: List of email texts

        Returns:
            List of prediction dictionaries with label and confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Vectorize
        X_vectorized = self.vectorizer.transform(texts)

        # Predict
        predictions = self.classifier.predict(X_vectorized)
        probabilities = self.classifier.predict_proba(X_vectorized)

        # Format results
        results = []
        for pred, probs in zip(predictions, probabilities):
            label = self.reverse_label_map[pred]
            confidence = float(max(probs))  # Confidence is the max probability

            results.append({
                'label': label,
                'is_relevant': label == 'relevant',
                'confidence': confidence,
                'probabilities': {
                    'not_relevant': float(probs[0]),
                    'relevant': float(probs[1])
                }
            })

        return results

    def save_model(self, model_path: str = "models/email_classifier.pkl"):
        """
        Save trained model to disk

        Args:
            model_path: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map
        }

        joblib.dump(model_data, model_path)
        print(f"\nModel saved to: {model_path}")

    def load_model(self, model_path: str = "models/email_classifier.pkl"):
        """
        Load trained model from disk

        Args:
            model_path: Path to load model from
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_data = joblib.load(model_path)

        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.label_map = model_data['label_map']
        self.reverse_label_map = model_data['reverse_label_map']
        self.is_trained = True

        print(f"Model loaded from: {model_path}")

    def get_top_features(self, n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top features for each class

        Args:
            n: Number of top features to return

        Returns:
            Dictionary mapping class names to top features
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        feature_names = np.array(self.vectorizer.get_feature_names_out())

        top_features = {}
        for class_idx, class_name in self.reverse_label_map.items():
            # Get feature log probabilities for this class
            log_probs = self.classifier.feature_log_prob_[class_idx]

            # Get top n features
            top_indices = np.argsort(log_probs)[-n:][::-1]
            top_features[class_name] = [
                (feature_names[i], log_probs[i]) for i in top_indices
            ]

        return top_features


if __name__ == "__main__":
    print("Model module loaded successfully!")
    print("Use train.py script to train the model")
