import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from lotus.dtype_extensions import convert_to_base_data
from lotus.models.rm import RM


class Model2VecRM(RM):
    """
    A retrieval model based on model2vec static embeddings.

    This class provides functionality to generate embeddings for documents using
    model2vec static models. Unlike transformer-based models, model2vec uses
    lookup-based embeddings which are much faster and don't require PyTorch.

    Attributes:
        model (str): Name of the model2vec model to use.
        max_batch_size (int): Maximum batch size for embedding requests.
        static_model (StaticModel): The model2vec static model instance.
    """

    def __init__(
        self,
        model: str = "minishlab/potion-base-8M",
        max_batch_size: int = 512,
    ) -> None:
        """
        Initialize the Model2VecRM retrieval model.

        Args:
            model: Name of the model2vec model to use.
                   Defaults to "minishlab/potion-base-8M".
            max_batch_size: Maximum batch size for embedding requests.
                           Defaults to 512 (higher than transformers since it's faster).
        """
        try:
            from model2vec import StaticModel
        except ImportError:
            raise ImportError(
                "The 'model2vec' library is required for Model2VecRM. "
                "You can install it with the following command:\n\n"
                "    pip install 'lotus-ai[model2vec]'"
            )

        self.model: str = model
        self.max_batch_size: int = max_batch_size
        self.static_model: StaticModel = StaticModel.from_pretrained(model)

    def _embed(self, docs: list[str]) -> NDArray[np.float64]:
        """
        Generate embeddings for a list of documents using model2vec.

        This method processes documents in batches to generate embeddings using
        the specified model2vec model. model2vec is significantly faster than
        transformer models as it uses static (lookup-based) embeddings.

        Args:
            docs: List of document strings to embed.

        Returns:
            NDArray[np.float64]: Array of embeddings with shape (num_docs, embedding_dim).

        Raises:
            Exception: If the embedding generation fails.
        """
        all_embeddings = []
        for i in tqdm(range(0, len(docs), self.max_batch_size), desc="Embedding with model2vec"):
            batch = docs[i : i + self.max_batch_size]
            _batch = convert_to_base_data(batch)
            # model2vec.encode returns numpy array directly
            embeddings = self.static_model.encode(_batch, show_progress_bar=False)
            # Ensure float64 dtype for consistency with other RM implementations
            embeddings = embeddings.astype(np.float64)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)
