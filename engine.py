from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from database import get_retriever

def get_qa_chain(model_name: str = "llama3"):
    llm = ChatOllama(model=model_name)
    retriever = get_retriever()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return chain

def ask_question(question: str, model_name: str = "llama3"):
    chain = get_qa_chain(model_name=model_name)
    result = chain.invoke({"query": question})
    return result
