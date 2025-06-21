from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile


def qa_agent(openai_api_key, memory, uploaded_files, question):
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    # file_content = uploaded_file.read()
    # temp_file_path = "temp.pdf"
    # with open(temp_file_path, "wb") as temp_file:
    #     temp_file.write(file_content)
    # loader = PyPDFLoader(temp_file_path)
    # docs = loader.load()
    all_docs = []

    if uploaded_files: 
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.read())
                temp_path = temp_file.name
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            all_docs.extend(docs)
    else:  # if user does not upload any files, load from the Solutions folder
        folder_path = "Solutions"
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(folder_path, filename)
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                all_docs.extend(docs)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(all_docs)
    embeddings_model = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = qa.invoke({"chat_history": memory, "question": question})
    return response
