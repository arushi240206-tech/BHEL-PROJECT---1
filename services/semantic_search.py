import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class SemanticSearchService:
    """
    Semantic search engine using Sentence Transformers (all-MiniLM-L6-v2)
    and FAISS for fast approximate nearest-neighbor retrieval.

    The TF-IDF engine in MLService is kept as a fallback — this service
    signals readiness via `self.ready` so callers can gracefully degrade.
    """

    MODEL_NAME = 'all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384

    def __init__(self, embeddings_dir='embeddings'):
        self.embeddings_dir = embeddings_dir
        self.index_path = os.path.join(embeddings_dir, 'faiss_index.bin')
        self.meta_path = os.path.join(embeddings_dir, 'corpus_metadata.json')
        self.model = None
        self.index = None
        self.corpus_indices = []  # Maps FAISS position → DataFrame row index
        self.ready = False

    # ------------------------------------------------------------------
    # Build (offline / one-time)
    # ------------------------------------------------------------------
    def build_embeddings(self, df):
        """
        Build FAISS index from the DataFrame.
        Concatenates the same 9 text columns used by the TF-IDF engine,
        encodes them with Sentence Transformers, normalizes, and stores.
        """
        print("[SemanticSearch] Loading sentence-transformer model...")
        self._ensure_model()

        corpus_columns = [
            'Problem Description',
            'Item',
            'Defect Type',
            'Defect Sub-type Description',
            'Problem Nature Keywords',
            'Product',
            'Project',
            'PGMA Description',
            'Equipment Name'
        ]

        # Build combined text corpus
        texts = df[corpus_columns].fillna('').agg(' '.join, axis=1).tolist()
        indices = df.index.tolist()

        print(f"[SemanticSearch] Encoding {len(texts)} documents...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=256,
            normalize_embeddings=True  # L2-normalize so inner product = cosine sim
        )
        embeddings = np.array(embeddings, dtype='float32')

        # Build FAISS inner-product index
        print("[SemanticSearch] Building FAISS index...")
        index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        index.add(embeddings)

        # Persist
        os.makedirs(self.embeddings_dir, exist_ok=True)
        faiss.write_index(index, self.index_path)
        with open(self.meta_path, 'w') as f:
            json.dump({'indices': indices, 'count': len(indices)}, f)

        self.index = index
        self.corpus_indices = indices
        self.ready = True
        print(f"[SemanticSearch] Index built and saved. {index.ntotal} vectors indexed.")

    # ------------------------------------------------------------------
    # Load (startup)
    # ------------------------------------------------------------------
    def load_index(self):
        """Load a previously-built FAISS index from disk."""
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            print("[SemanticSearch] No pre-built index found. Semantic search unavailable (TF-IDF fallback active).")
            return False

        try:
            self._ensure_model()
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'r') as f:
                meta = json.load(f)
            self.corpus_indices = meta['indices']
            self.ready = True
            print(f"[SemanticSearch] FAISS index loaded. {self.index.ntotal} vectors ready.")
            return True
        except Exception as e:
            print(f"[SemanticSearch] Failed to load index: {e}")
            self.ready = False
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query, top_k=10):
        """
        Encode the query and perform nearest-neighbor search.

        Returns:
            list of (dataframe_row_index, similarity_score) tuples,
            sorted descending by score.
        """
        if not self.ready or self.index is None:
            return []

        self._ensure_model()

        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype('float32')

        scores, positions = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0 or pos >= len(self.corpus_indices):
                continue
            results.append((self.corpus_indices[pos], float(score)))

        return results

    # ------------------------------------------------------------------
    # Rebuild (after new cases are merged)
    # ------------------------------------------------------------------
    def rebuild(self, df):
        """Convenience wrapper — same as build_embeddings but for incremental updates."""
        self.build_embeddings(df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _ensure_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.MODEL_NAME)
