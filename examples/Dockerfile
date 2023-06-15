FROM python:3.10

WORKDIR /langchain

RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="${PATH}:/root/.local/bin"

RUN git clone --branch gateway-integration https://github.com/harupy/langchain.git /langchain
RUN poetry install
