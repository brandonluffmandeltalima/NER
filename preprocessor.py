"""
Text Preprocessing Module
Cleans and prepares email text for machine learning
"""
import re
import string
from typing import List


class TextPreprocessor:
    """Preprocesses email text for classification"""

    def __init__(self, lowercase: bool = True, remove_numbers: bool = False):
        """
        Initialize preprocessor

        Args:
            lowercase: Convert text to lowercase
            remove_numbers: Remove numeric characters
        """
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers

        # Common legal/email stop words to remove (optional, can improve accuracy)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text

        Args:
            text: Raw text string

        Returns:
            Cleaned text string
        """
        if not text:
            return ""

        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove HTML tags (if any)
        text = re.sub(r'<[^>]+>', '', text)

        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)

        # Remove extra punctuation
        text = re.sub(r'[^\w\s\-]', ' ', text)

        # Optionally remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove extra spaces
        text = ' '.join(text.split())

        return text.strip()

    def remove_stop_words(self, text: str) -> str:
        """
        Remove common stop words

        Args:
            text: Text string

        Returns:
            Text with stop words removed
        """
        words = text.split()
        filtered_words = [w for w in words if w.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def preprocess(self, text: str, remove_stops: bool = False) -> str:
        """
        Full preprocessing pipeline

        Args:
            text: Raw text
            remove_stops: Whether to remove stop words

        Returns:
            Preprocessed text
        """
        text = self.clean_text(text)

        if remove_stops:
            text = self.remove_stop_words(text)

        return text

    def preprocess_email(self, subject: str, body: str, remove_stops: bool = False) -> str:
        """
        Preprocess email (subject + body combined)

        Args:
            subject: Email subject
            body: Email body
            remove_stops: Whether to remove stop words

        Returns:
            Combined preprocessed text
        """
        # Combine subject and body (subject is weighted by appearing twice)
        combined = f"{subject} {subject} {body}"
        return self.preprocess(combined, remove_stops=remove_stops)


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()

    test_subject = "Case 2024-CM-1007 - Investigation Status Update"
    test_body = """
    MAJ Sarah Chen,

    Providing update on 2024-CM-1007 (Article 120 UCMJ - Sexual Assault).

    Visit https://example.com for more info.
    Contact: michael.rodriguez@jag.mil

    DEFENDANT: SSG Michael Brown
    """

    print("Original subject:", test_subject)
    print("\nOriginal body:", test_body)

    cleaned = preprocessor.preprocess_email(test_subject, test_body)
    print("\nCleaned text:", cleaned)

    cleaned_no_stops = preprocessor.preprocess_email(test_subject, test_body, remove_stops=True)
    print("\nCleaned (no stop words):", cleaned_no_stops)
