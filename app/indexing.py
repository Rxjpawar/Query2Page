from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore

load_dotenv()

#file loading
pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(file_path=pdf_path)
documents = loader.load() #read pdf file


#text splitting / chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)
split_documents = text_splitter.split_documents(documents=documents)


#Hugging Face embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': False}
)


# Create vector store
vector_store = QdrantVectorStore.from_documents(
    documents=split_documents,
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embedding_model,
)

print("Indexing of documents is complete")