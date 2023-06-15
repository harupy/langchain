from langchain.embeddings import MlflowGatewayEmbeddings


embeddings = MlflowGatewayEmbeddings(
    gateway_uri="http://127.0.0.1:5000",
    route="embeddings",
)


print(embeddings.embed_query("foo"))
print(embeddings.embed_documents(["foo"]))
