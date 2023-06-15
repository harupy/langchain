## Setup

```sh
# Clone this PR branch
git clone --branch gateway-integration https://github.com/harupy/langchain.git
cd langchain

# Install dependencies
poetry install

# Set an OpenAI API key
export OPENAI_API_KEY=...

# Run a gateway server
poetry run mlflow gateway start --config-path examples/config.yaml
```

Once the gateway server starts running, open another terminal and run the following commands:

## Running Examples

```sh
poetry run python examples/completions.py
poetry run python examples/embeddings.py
```

## Docker (preferred)

All you need is Docker.

```sh
git clone --branch gateway-integration https://github.com/harupy/langchain.git
cd langchain

export OPENAI_API_KEY=...
docker build -t mlflow-langchain examples
docker run -it --rm -e OPENAI_API_KEY="$OPENAI_API_KEY" --name mlflow-langchain mlflow-langchain bash
poetry run mlflow gateway start --config-path examples/config.yaml
```

On another terminal:

```sh
docker exec -it mlflow-langchain bash
poetry run python examples/completions.py
poetry run python examples/embeddings.py
```

# Formatting

```sh
poetry run black .
```
