from __future__ import annotations
from functools import lru_cache
from typing import List, Any, Optional,TYPE_CHECKING

from pydantic import BaseModel

from langchain.embeddings.base import Embeddings


if TYPE_CHECKING:
    import mlflow.gateway


@lru_cache()
def _get_client(gateway_uri: Optional[str]) -> mlflow.gateway.MlflowGatewayClient:
    import mlflow.gateway

    return mlflow.gateway.MlflowGatewayClient(gateway_uri)


class MlflowGatewayEmbeddings(Embeddings, BaseModel):
    route: str
    gateway_uri: Optional[str]

    def __init__(self, **kwargs: Any):
        import mlflow

        super().__init__(**kwargs)
        mlflow.gateway.set_gateway_uri(self.gateway_uri)

    def _query(self, texts: List[str]) -> List[List[float]]:
        resp = _get_client(self.gateway_uri).query(
            self.route,
            data={"text": texts},
        )
        return resp["embeddings"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._query(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._query([text])[0]
