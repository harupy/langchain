from __future__ import annotations

from typing import Any, List, Dict, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import mlflow.gateway


class MlflowGateway(LLM):
    gateway_uri: str
    route: str
    temperature: float = 0.0
    stop: List[str] | None = None
    max_tokens: Optional[int] = None
    candidate_count: int = 5

    def __init__(self, **kwargs: Any):
        import mlflow

        super().__init__(**kwargs)
        mlflow.gateway.set_gateway_uri(self.gateway_uri)

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "gateway_uri": self.gateway_uri,
            "route": self.route,
            "temperature": self.temperature,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "candidate_count": self.candidate_count,
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
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "candidate_count": self.candidate_count,
        }
        stop = stop or self.stop
        if stop:
            data["stop"] = stop
        resp = mlflow.gateway.query(self.route, data=data)
        return resp["candidates"][0]["text"]

    @property
    def _llm_type(self) -> str:
        return "mlflow-gateway"
