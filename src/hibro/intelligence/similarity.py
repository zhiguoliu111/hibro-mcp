#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Similarity Calculation Module
Calculates text semantic similarity using local sentence-transformers model
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import hashlib

# Lazy import sentence_transformers (avoid loading at startup)
# from sentence_transformers import SentenceTransformer
SENTENCE_TRANSFORMERS_AVAILABLE = None  # Lazy detection

from ..utils.helpers import sanitize_content


def _check_sentence_transformers():
    """Lazy check if sentence_transformers is available"""
    global SENTENCE_TRANSFORMERS_AVAILABLE
    if SENTENCE_TRANSFORMERS_AVAILABLE is None:
        try:
            from sentence_transformers import SentenceTransformer
            SENTENCE_TRANSFORMERS_AVAILABLE = True
        except ImportError:
            SENTENCE_TRANSFORMERS_AVAILABLE = False
    return SENTENCE_TRANSFORMERS_AVAILABLE


class SimilarityCalculator:
    """Semantic similarity calculator"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Optional[str] = None):
        """
        Initialize similarity calculator

        Args:
            model_name: Model name to use
            cache_dir: Cache directory
        """
        self.logger = logging.getLogger('hibro.similarity_calculator')
        self.model_name = model_name

        # Set cache directory
        if cache_dir is None:
            cache_dir = Path.home() / '.hibro' / 'similarity_cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = None
        self._embedding_cache = {}
        self._load_model()
        self._load_cache()

    def _load_model(self):
        """Load sentence-transformers model"""
        # Lazy check and import
        if not _check_sentence_transformers():
            self.logger.warning("sentence-transformers not installed, will use simple text similarity algorithm")
            return

        try:
            # Lazy import
            from sentence_transformers import SentenceTransformer
            self.logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Model loading completed")
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            self.model = None

    def _load_cache(self):
        """Load embedding vector cache"""
        cache_file = self.cache_dir / f'embeddings_{self.model_name.replace("/", "_")}.pkl'

        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self._embedding_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self._embedding_cache)} cached embedding vectors")
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            self._embedding_cache = {}

    def _save_cache(self):
        """Save embedding vector cache"""
        cache_file = self.cache_dir / f'embeddings_{self.model_name.replace("/", "_")}.pkl'

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._embedding_cache, f)
            self.logger.debug(f"Saved {len(self._embedding_cache)} embedding vectors to cache")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def _get_text_hash(self, text: str) -> str:
        """Get text hash for caching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            return np.zeros(384)  # Default dimension for MiniLM model

        # Clean text
        cleaned_text = sanitize_content(text.strip())
        text_hash = self._get_text_hash(cleaned_text)

        # Check cache
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]

        # Calculate embedding vector
        if self.model is not None:
            try:
                embedding = self.model.encode(cleaned_text, convert_to_numpy=True)
                # Cache result
                self._embedding_cache[text_hash] = embedding
                return embedding
            except Exception as e:
                self.logger.error(f"Failed to calculate embedding vector: {e}")
                return self._fallback_embedding(cleaned_text)
        else:
            return self._fallback_embedding(cleaned_text)

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        Fallback embedding vector calculation (simple TF-IDF style)

        Args:
            text: Input text

        Returns:
            Simple vector representation
        """
        # Simple character-level features
        words = text.lower().split()

        # Create a fixed-dimension vector
        embedding = np.zeros(384)

        for i, word in enumerate(words[:100]):  # Limit word count
            # Use word hash as feature
            word_hash = hash(word) % 384
            embedding[word_hash] += 1.0 / (i + 1)  # Position weight

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts

        Args:
            text1: Text 1
            text2: Text 2

        Returns:
            Similarity score (0.0-1.0)
        """
        if not text1 or not text2:
            return 0.0

        # Get embedding vectors
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)

        # Calculate cosine similarity
        similarity = self._cosine_similarity(embedding1, embedding2)

        # Ensure result is in 0-1 range
        return max(0.0, min(1.0, similarity))

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: Vector 1
            vec2: Vector 2

        Returns:
            Cosine similarity
        """
        # Calculate dot product
        dot_product = np.dot(vec1, vec2)

        # Calculate vector norms
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = dot_product / (norm1 * norm2)

        # Map result from [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    def find_similar_texts(self, query_text: str, candidate_texts: List[str],
                          top_k: int = 5, min_similarity: float = 0.3) -> List[Tuple[str, float]]:
        """
        Find most similar texts to query text in candidates

        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Return top k most similar
            min_similarity: Minimum similarity threshold

        Returns:
            List of (text, similarity score) sorted by similarity descending
        """
        if not query_text or not candidate_texts:
            return []

        # Calculate query text embedding vector
        query_embedding = self.get_embedding(query_text)

        # Calculate similarity for all candidate texts
        similarities = []
        for text in candidate_texts:
            if text and text.strip():
                text_embedding = self.get_embedding(text)
                similarity = self._cosine_similarity(query_embedding, text_embedding)

                if similarity >= min_similarity:
                    similarities.append((text, similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        return similarities[:top_k]

    def batch_similarity(self, texts: List[str]) -> np.ndarray:
        """
        Calculate similarity matrix between texts in batch

        Args:
            texts: List of texts

        Returns:
            Similarity matrix (n x n)
        """
        if not texts:
            return np.array([])

        # Get embedding vectors for all texts
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(texts), len(texts)))

        for i in range(len(texts)):
            for j in range(len(texts)):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                    similarity_matrix[i][j] = similarity

        return similarity_matrix

    def cluster_similar_texts(self, texts: List[str], similarity_threshold: float = 0.7) -> List[List[int]]:
        """
        Cluster texts based on similarity

        Args:
            texts: List of texts
            similarity_threshold: Similarity threshold

        Returns:
            Clustering result, each cluster contains text index list
        """
        if not texts:
            return []

        # Calculate similarity matrix
        similarity_matrix = self.batch_similarity(texts)

        # Simple clustering algorithm
        clusters = []
        used_indices = set()

        for i in range(len(texts)):
            if i in used_indices:
                continue

            # Create new cluster
            cluster = [i]
            used_indices.add(i)

            # Find similar texts
            for j in range(i + 1, len(texts)):
                if j in used_indices:
                    continue

                if similarity_matrix[i][j] >= similarity_threshold:
                    cluster.append(j)
                    used_indices.add(j)

            clusters.append(cluster)

        self.logger.info(f"Clustered {len(texts)} texts into {len(clusters)} clusters")
        return clusters

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Cache statistics
        """
        return {
            'cache_size': len(self._embedding_cache),
            'model_name': self.model_name,
            'model_available': self.model is not None,
            'cache_dir': str(self.cache_dir)
        }

    def clear_cache(self):
        """Clear cache"""
        self._embedding_cache.clear()
        self.logger.info("Embedding vector cache cleared")

    def save_cache_to_disk(self):
        """Manually save cache to disk"""
        self._save_cache()

    def __del__(self):
        """Destructor, save cache"""
        try:
            self._save_cache()
        except:
            pass  # Ignore errors during destruction


class SemanticSearchEngine:
    """Semantic search engine"""

    def __init__(self, similarity_calculator: SimilarityCalculator):
        """
        Initialize semantic search engine

        Args:
            similarity_calculator: Similarity calculator
        """
        self.similarity_calc = similarity_calculator
        self.logger = logging.getLogger('hibro.semantic_search')

    def search_memories(self, query: str, memories: List[Dict[str, Any]],
                       top_k: int = 10, min_similarity: float = 0.3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform semantic search in memories

        Args:
            query: Search query
            memories: List of memories
            top_k: Return top k results
            min_similarity: Minimum similarity threshold

        Returns:
            List of (memory, similarity score)
        """
        if not query or not memories:
            return []

        # Extract memory content
        memory_contents = [memory.get('content', '') for memory in memories]

        # Perform similarity search
        similar_results = self.similarity_calc.find_similar_texts(
            query, memory_contents, top_k, min_similarity
        )

        # Match back to original memory objects
        results = []
        for content, similarity in similar_results:
            for memory in memories:
                if memory.get('content') == content:
                    results.append((memory, similarity))
                    break

        self.logger.info(f"Semantic search '{query}' returned {len(results)} results")
        return results

    def find_related_memories(self, target_memory: Dict[str, Any],
                            candidate_memories: List[Dict[str, Any]],
                            top_k: int = 5, min_similarity: float = 0.4) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find memories related to target memory

        Args:
            target_memory: Target memory
            candidate_memories: List of candidate memories
            top_k: Return top k results
            min_similarity: Minimum similarity threshold

        Returns:
            List of related memories
        """
        target_content = target_memory.get('content', '')

        return self.search_memories(
            target_content, candidate_memories, top_k, min_similarity
        )

    def group_similar_memories(self, memories: List[Dict[str, Any]],
                             similarity_threshold: float = 0.6) -> List[List[Dict[str, Any]]]:
        """
        Group similar memories

        Args:
            memories: List of memories
            similarity_threshold: Similarity threshold

        Returns:
            List of memory groups
        """
        if not memories:
            return []

        # Extract memory content
        memory_contents = [memory.get('content', '') for memory in memories]

        # Perform clustering
        clusters = self.similarity_calc.cluster_similar_texts(
            memory_contents, similarity_threshold
        )

        # Convert indices back to memory objects
        memory_groups = []
        for cluster in clusters:
            group = [memories[i] for i in cluster]
            memory_groups.append(group)

        self.logger.info(f"Grouped {len(memories)} memories into {len(memory_groups)} groups")
        return memory_groups