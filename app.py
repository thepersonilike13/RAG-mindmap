import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import matplotlib.pyplot as plt
import networkx as nx
import tempfile

# Load environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Mind Map Drawing Function
def draw_mind_map(data: dict, title: str = "Mind Map"):
    G = nx.Graph()

    def add_edges(parent, children):
        for child in children:
            if isinstance(child, dict):
                for sub_parent, sub_children in child.items():
                    G.add_edge(parent, sub_parent)
                    add_edges(sub_parent, sub_children)
            else:
                G.add_edge(parent, child)

    for root, branches in data.items():
        add_edges(root, branches)

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(
        G, pos, with_labels=True, node_color="skyblue", node_size=2000,
        font_size=10, font_weight='bold', edge_color="gray", ax=ax
    )
    plt.title(title, fontsize=16)
    plt.axis("off")
    st.pyplot(fig)

# PDF Reading and Text Extraction
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Text Chunking
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Vector Store Creation
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(chunks, embedding=embeddings)
    store.save_local("faiss_index")

# Conversational QA Chain
def get_conversational_chain():
    template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not available, just say: "answer is not available in the context".

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Key-Value Pair Chain
import ast
import re

def extract_key_value_pairs(text):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    prompt = f"""
    Convert the following answer into a clean Python dictionary (key-value format), 
    suitable for a mind map structure. Use plain Python syntax only, no markdown or explanations.

    Text:\n{text}\n
    Output:
    """
    response = model.invoke(prompt)

    # Safely extract dictionary-looking part
    if hasattr(response, "content"):
        response_text = response.content
    else:
        response_text = str(response)

    # Remove markdown-like formatting if present (e.g., ```python ... ```)
    response_text = re.sub(r"```(?:python)?", "", response_text).strip("` \n")

    try:
        result = ast.literal_eval(response_text)
        if isinstance(result, dict):
            return result
        else:
            raise ValueError("Parsed response is not a dictionary.")
    except Exception as e:
        st.error(f"❌ Failed to parse mind map: {e}")
        st.code(response_text, language="python")
        return {}

# User Question Handler
def user_input_handler(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response["output_text"]
    st.subheader("📄 Answer:")
    st.write(answer)

    # Convert to Key-Value
    mind_map_data = extract_key_value_pairs(answer)
    if mind_map_data:
        st.subheader("🧠 Mind Map:")
        draw_mind_map(mind_map_data, title="Answer Mind Map")

# Main Streamlit UI
def main():
    st.set_page_config(page_title="LLM PDF Chat + Mind Map")
    st.title("📚 PDF Q&A + Mind Map Generator")

    user_question = st.text_input("🔍 Ask a Question from the PDF")

    with st.sidebar:
        st.header("📁 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("📦 Submit & Process") and pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ PDF Processed")

    if user_question:
        user_input_handler(user_question)

if __name__ == "__main__":
    main()
