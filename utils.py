from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from api import GEMINI_API_KEY
from langchain_community.document_loaders import UnstructuredURLLoader



def docs_loader(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    return data



def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000)
    chunks = text_splitter.split_documents(data)
    return chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings 
def creat_db(chunks , embeddings):
    vectorstore = Chroma.from_documents(documents= chunks , embedding=embeddings)
    retriver = vectorstore.as_retriever(search_type = 'similarity'  , search_kwargs = {'k' : 3})
    return retriver

def propmt_completion(llm , messsage , retriever):
    system_prompt = (
    "You are helful assistant for question answering task."
    "Use the following pieces of retrived context to answer "
    "the question. if you don't know the answer say that you"
    "don't know. Use three sentences maximum and keep the "
    "answer concise"
    "\n\n"
    "{context}"
)


    prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , system_prompt),
        ("human" , "{input}"),
    ]
)
    questionAnswerChain = create_stuff_documents_chain(llm , prompt)
    rag_chain = create_retrieval_chain(retriever , questionAnswerChain)
    response = rag_chain.invoke({'input' : f"{messsage}"})
    answer = response['answer']
    return answer
