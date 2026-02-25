"""TF-IDF keyword search baseline for comparison with semantic search."""
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KeywordSearcher:
    """Simple TF-IDF search over a directory of markdown files."""

    def __init__(self, docs_path: str):
        self.docs_path = docs_path
        self.filenames = []
        self.contents = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self):
        """Read all markdown files and build a TF-IDF index."""
        path = Path(self.docs_path)
        for filepath in sorted(path.glob("**/*.md")):
            text = filepath.read_text(encoding="utf-8")
            self.filenames.append(filepath.name)
            self.contents.append(text)

        for filepath in sorted(path.glob("**/*.mdx")):
            text = filepath.read_text(encoding="utf-8")
            self.filenames.append(filepath.name)
            self.contents.append(text)

        if not self.contents:
            raise ValueError(f"No markdown files found in {self.docs_path}")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.contents)

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        """Search for documents matching a query.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of (filename, score) tuples, sorted by score descending.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        ranked = sorted(
            zip(self.filenames, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python keyword_baseline.py <docs_path> <query>")
        sys.exit(1)

    searcher = KeywordSearcher(sys.argv[1])
    results = searcher.search(" ".join(sys.argv[2:]))
    for filename, score in results:
        print(f"{score:.4f}  {filename}")
