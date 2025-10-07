import os
import pickle
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lotus.types import RMOutput
from lotus.vector_store.vs import VS


class VicinityVS(VS):
    """
    Vector store using Vicinity library with flexible backends.

    Vicinity is a lightweight vector store from MinishLab (same creators as model2vec).
    It provides a unified interface across multiple backends:
    - BASIC: Pure Python exact search (no external deps)
    - HNSW: Fast approximate search (requires vicinity[hnsw])
    - FAISS: Multiple FAISS indexes (requires vicinity[faiss])
    - ANNOY, USEARCH, VOYAGER: Other ANN backends

    For most use cases, BASIC backend is sufficient and has zero dependencies.
    """

    def __init__(self, backend: str = "BASIC", **backend_params: Any) -> None:
        """
        Initialize VicinityVS.

        Args:
            backend: Backend to use. Options: BASIC, HNSW, FAISS, ANNOY, USEARCH, VOYAGER, PYNNDESCENT
                     Default is BASIC (pure Python, no external deps).
            **backend_params: Additional parameters for the backend.
                             For BASIC: metric (default: "cosine")
                             For HNSW: metric, ef_construction, m
                             See Vicinity documentation for full parameter list.
        """
        super().__init__()
        try:
            from vicinity import Backend, Metric, Vicinity
        except ImportError:
            raise ImportError(
                "The 'vicinity' library is required for VicinityVS. "
                "You can install it with the following command:\n\n"
                "    pip install 'lotus-ai[model2vec]'"
            )

        self.backend = backend
        self.backend_params = backend_params
        self.vicinity: Vicinity | None = None
        self.items: list[int] | None = None  # Store original indices as items
        self.Backend = Backend
        self.Metric = Metric
        self.Vicinity = Vicinity

    def index(self, docs: list[str], embeddings: NDArray[np.float64], index_dir: str, **kwargs: dict[str, Any]) -> None:
        """
        Create index and store it in the vector store.

        Args:
            docs: List of document strings (not used, only indices matter).
            embeddings: Document embeddings.
            index_dir: Directory to save the index.
            **kwargs: Additional parameters.
        """
        # Store indices as items for later retrieval
        self.items = list(range(len(embeddings)))

        # Get backend type enum
        try:
            backend_type = getattr(self.Backend, self.backend)
        except AttributeError:
            raise ValueError(
                f"Unknown backend: {self.backend}. "
                f"Available backends: {[b.name for b in self.Backend]}"
            )

        # Create Vicinity instance
        self.vicinity = self.Vicinity.from_vectors_and_items(
            vectors=embeddings,
            items=self.items,
            backend_type=backend_type,
            **self.backend_params,
        )

        # Save to disk
        # Don't create the directory - Vicinity will do it
        # But ensure parent directory exists
        parent_dir = os.path.dirname(index_dir)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        self.vicinity.save(index_dir, overwrite=True)

        # Also save vectors separately for get_vectors_from_index
        # Vicinity doesn't expose internal vectors, so we store them ourselves
        with open(f"{index_dir}/vectors.pkl", "wb") as f:
            pickle.dump(embeddings, f)

        self.index_dir = index_dir

    def load_index(self, index_dir: str) -> None:
        """
        Load the index from the vector store into memory.

        Args:
            index_dir: Directory containing the saved index.
        """
        self.index_dir = index_dir
        self.vicinity = self.Vicinity.load(index_dir)
        # Items are restored from the saved index
        self.items = self.vicinity.items

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        """
        Retrieve vectors from a stored index given specific ids.

        Args:
            index_dir: Directory containing the saved index.
            ids: List of document ids to retrieve.

        Returns:
            Array of vectors corresponding to the ids.
        """
        # Load vectors from pickle file
        with open(f"{index_dir}/vectors.pkl", "rb") as f:
            all_vectors: NDArray[np.float64] = pickle.load(f)

        return all_vectors[ids]

    def __call__(
        self, query_vectors: NDArray[np.float64], K: int, ids: list[int] | None = None, **kwargs: dict[str, Any]
    ) -> RMOutput:
        """
        Perform a nearest neighbor search given query vectors.

        Args:
            query_vectors: Query vector(s) for the search.
            K: Number of nearest neighbors to retrieve.
            ids: Optional list of document ids to search within.
                 If None, search across all indexed vectors.
            **kwargs: Additional parameters.

        Returns:
            RMOutput containing distances and indices.
        """
        if self.vicinity is None:
            raise ValueError("Index not loaded. Call load_index() or index() first.")

        if ids is not None:
            # Filter to subset of vectors
            subset_vectors = self.get_vectors_from_index(self.index_dir, ids)  # type: ignore

            # Create temporary Vicinity instance for the subset
            backend_type = getattr(self.Backend, self.backend)
            temp_vicinity = self.Vicinity.from_vectors_and_items(
                vectors=subset_vectors,
                items=ids,  # Use original ids as items
                backend_type=backend_type,
                **self.backend_params,
            )

            # Query the subset
            results = temp_vicinity.query(query_vectors, k=K)

            # Results format: list of lists of (item, distance) tuples
            # Convert to the RMOutput format
            # NOTE: Vicinity returns distances, but LOTUS expects similarities
            # For cosine: similarity = 1 - distance
            indices = []
            distances = []
            for query_result in results:
                query_indices = [item for item, _ in query_result]
                query_distances = [1.0 - dist for _, dist in query_result]  # Convert distance to similarity
                indices.append(query_indices)
                distances.append(query_distances)

            return RMOutput(distances=np.array(distances), indices=np.array(indices))
        else:
            # Query the full index
            results = self.vicinity.query(query_vectors, k=K)

            # Convert Vicinity results to RMOutput format
            # NOTE: Vicinity returns distances, but LOTUS expects similarities
            # For cosine: similarity = 1 - distance
            indices = []
            distances = []
            for query_result in results:
                query_indices = [item for item, _ in query_result]
                query_distances = [1.0 - dist for _, dist in query_result]  # Convert distance to similarity
                indices.append(query_indices)
                distances.append(query_distances)

            return RMOutput(distances=np.array(distances), indices=np.array(indices))
