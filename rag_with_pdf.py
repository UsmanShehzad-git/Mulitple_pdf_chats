from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pdfplumber
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import io

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
 

#Define the chunks of PDF Text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000 , chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#Define the Vector Store and Initialize the Embeddings
def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text chunks found. Cannot create FAISS index.")
        return
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not embeddings:
        st.error("No embeddings found. Cannot create FAISS index.")
        return
    vector_store = FAISS.from_texts(text_chunks ,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = '''
    Answer the question as detailed as possible from the provided context, make sure to provide all the detail, if the answer is not in
    provided context just say "Answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer :
    '''
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt = PromptTemplate(template=prompt_template , input_variables=["context","question"])
    chain = load_qa_chain(model ,chain_type="stuff", prompt=prompt)
    return chain

#Define the Useru prespectives
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index" , embeddings , allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs , "question": user_question}
        ,return_only_outputs=True )
    
    print(response)
    st.write("Response :\n",response["output_text"])

def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat with Multiple PDF using Gemini Pro")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process", type="pdf")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                with pdfplumber.open(io.BytesIO(pdf_docs.getvalue())) as pdf:
                  text = ""
                  for page in pdf.pages:
                     text += page.extract_text()

                raw_text = text
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()