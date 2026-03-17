import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from util.embeddings import get_embeddings
from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input

from typing import Annotated
from pydantic import Field


def load_documents(directory_path: str):
    """Load and index all documents from a directory."""
    if not os.path.exists(directory_path):
        return None

    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )

    docs = loader.load()
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(docs)

    embeddings = get_embeddings()
    return FAISS.from_documents(splits, embeddings)


def run():
    # Load documents into vectorstore
    documents_path = os.path.join(os.getcwd(), "documents")
    vector_store = load_documents(documents_path)

    @tool(response_format="content_and_artifact")
    def search_documents(query: Annotated[str, Field(description="Sökfråga")]):
        """Sök i de indexerade dokumenten för att hitta relevant information."""
        if vector_store is None:
            return "Inga dokument har laddats.", []

        retrieved_docs = vector_store.similarity_search(query, k=3)
        if not retrieved_docs:
            return "Hittade ingen relevant information.", []

        serialized = "\n\n".join(
            f"Källa: {doc.metadata.get('source', 'Okänd')}\nInnehåll: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    # Get predefined attributes
    model = get_model()

    # Create agent
    agent = create_agent(
        model=model,
        tools=[search_documents],
        system_prompt=(
            "Du är en hjälpsam assistent som kan söka i och svara på frågor om dokument. "
            "När användaren ställer en fråga, använd search_documents verktyget för att hitta relevant information. "
            "Basera dina svar på informationen från dokumenten. "
            "Om du inte hittar relevant information, säg det tydligt. "
            "Svara alltid på svenska och var koncis men informativ."
        )
    )

    # Get user input
    user_input = get_user_input("Ställ din fråga")

    # Call the agent
    process_stream = agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode=STREAM_MODES,
    )

    # Stream the process
    handle_stream(process_stream)


if __name__ == "__main__":
    run()
