from __future__ import annotations

from typing import List, Any

from pydantic import BaseModel

from langchain.embeddings.base import Embeddings


class MlflowGatewayEmbeddings(Embeddings, BaseModel):
    gateway_uri: str
    route: str

    def __init__(self, **kwargs: Any):
        import mlflow

        super().__init__(**kwargs)
        mlflow.gateway.set_gateway_uri(self.gateway_uri)

    def _query(self, texts: List[str]) -> List[float]:
        import mlflow

        resp = mlflow.gateway.query(
            self.route,
            data={"text": texts},
        )
        return resp["embeddings"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._query(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._query([text])[0]
