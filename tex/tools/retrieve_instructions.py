from langchain_core.vectorstores import InMemoryVectorStore

from tex.agents.schemas import FormInput
from tex.model import call_gemini_embedding

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(call_gemini_embedding)


def retrieve_instructions(state: FormInput):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}
