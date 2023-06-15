from langchain import LLMChain, PromptTemplate
from langchain.llms import MlflowGateway

gateway_llm = MlflowGateway(
    gateway_uri="http://127.0.0.1:5000",
    route="completions",
    temperature=0.3,
)

llm_chain = LLMChain(
    llm=gateway_llm,
    prompt=PromptTemplate(input_variables=["adjective"], template="I am a {adjective}"),
)
result = llm_chain.run(adjective="professional")
print(result)
