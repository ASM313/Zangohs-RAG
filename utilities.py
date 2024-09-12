from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

# Set up enenvironment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_key=os.getenv("GOOGLE_API_KEY")

# Task 1: Extract text from pdf
def extract_pdf(uploaded_pdf):
    pdf_content = ""
    for pdf in uploaded_pdf:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_content += page.extract_text()
    return pdf_content

# Task 2: Divide data into chunks
# chunk_size will be 10,000

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Task 3: Create embeddings through GoogleGenerativeAIEmbeddings

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=gemini_key, model = "models/embedding-001")
    # create embeddings of chunks and save it locally
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Task 4: Create llm instance and provide prompt
def llm_instance(user_question):

    prompt_template = """
    Answer all questions as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in this context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    and you should answer in the following format:\n
    Answer:
    """

    client = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(client, chain_type="stuff", prompt=prompt)

    return chain

# Task 5: process users query
def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=gemini_key)

    faiss_indexing = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = faiss_indexing.similarity_search(question)

    chain = llm_instance(question)


    response = chain(
        {"input_documents":docs, "question": question}
        , return_only_outputs=True)
    return response['output_text']

print("Everything fine")
