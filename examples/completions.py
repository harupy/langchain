import mlflow
from langchain import LLMChain, PromptTemplate
from langchain.llms import MlflowGateway

gateway_llm = MlflowGateway(
    gateway_uri="http://127.0.0.1:5000",
    route="completions",
    params={
        "temperature": 0.0,
        "top_p": 0.1,
    },
)

llm_chain = LLMChain(
    llm=gateway_llm,
    prompt=PromptTemplate(input_variables=["adjective"], template="I am a {adjective}"),
)
result = llm_chain.run(
    adjective="professional",
)
print(result)


with mlflow.start_run():
    info = mlflow.langchain.log_model(llm_chain, "model")

model = mlflow.langchain.load_model(info.model_uri)
print(model.run(adjective="professional"))
