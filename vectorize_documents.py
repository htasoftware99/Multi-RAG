from langchain_community.document_loaders import UnstructuredAPIFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# API anahtarını buraya ekleyin
api_key = "unstructured_key"

embeddings = HuggingFaceEmbeddings()

# UnstructuredAPIFileLoader API anahtarıyla başlatma
loader = DirectoryLoader(
    path="data", 
    glob="./*.pdf", 
    loader_cls=lambda p: UnstructuredAPIFileLoader(p, api_key=api_key)  # API anahtarını burada ekliyoruz
)

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
text_chunks = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory="vector_db")
print("Documents Vectorized")
