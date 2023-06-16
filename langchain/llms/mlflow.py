from __future__ import annotations
from functools import lru_cache
from typing import Any, List, Dict, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import BaseModel, Extra
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import mlflow.gateway


@lru_cache()
def _client(gateway_uri: str) -> mlflow.gateway.MlflowGatewayClient:
    import mlflow.gateway

    return mlflow.gateway.MlflowGatewayClient(gateway_uri)


class Params(BaseModel, extra=Extra.allow):
    temperature: float = 0.0
    stop: List[str] | None = None
    max_tokens: Optional[int] = None
    candidate_count: int = 5


class MlflowGateway(LLM):
    gateway_uri: str
    route: str
    params: Params | None = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "gateway_uri": self.gateway_uri,
            "route": self.route,
            "params": self.params.dict(),
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return self._default_params

    def _call(
        self,
        prompt: str,
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        data = {
            "prompt": prompt,
            **self.params.dict(exclude_none=True),
        }
        if s := stop or self.params.stop:
            data["stop"] = s
        resp = _client(self.gateway_uri).query(self.route, data=data)
        return resp["candidates"][0]["text"]

    @property
    def _llm_type(self) -> str:
        return "mlflow-gateway"
