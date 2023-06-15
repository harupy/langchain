from __future__ import annotations

from typing import Any, List
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import mlflow.gateway


class MlflowGateway(LLM):
    gateway_uri: str
    route: str
    temperature: float = 0.3

    def __init__(self, **kwargs: Any):
        import mlflow

        super().__init__(**kwargs)
        mlflow.gateway.set_gateway_uri(self.gateway_uri)

    def _call(
        self,
        prompt: str,
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        resp = mlflow.gateway.query(
            self.route,
            data={
                "temperature": self.temperature,
                "prompt": prompt,
            },
        )
        return resp["candidates"][0]["text"]

    def _llm_type(self) -> str:
        return "mlflow-gateway"
