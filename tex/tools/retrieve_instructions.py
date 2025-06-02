from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings  # noqa

from tex.agents.schemas import FormInput

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)


def retrieve_instructions(state: FormInput):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}
