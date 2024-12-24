import os
import re
import json
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import time

history_file_path = "./chat_history.json"

def save_history_to_file(history, file_path=history_file_path):
    with open(file_path, "w") as f:
        json.dump(history, f)

def load_history_from_file(file_path=history_file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []

os.environ["OPENAI_API_KEY"] = "Input_Your_OpenAI_API_KEY"

system_message_content = """당신은 정보처리기사 실기 기출문제 답변을 제공하는 AI 어시스턴트입니다.
기출문제의 출제 항목을 바탕으로 질문에 답변하세요.
답변이 없을 경우, '해당 정보는 존재하지 않습니다.'라고 안내하세요."""

# Streamlit 스타일 적용
st.markdown("""
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 20px;
    }
    .question-box {
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        background-color: #f9f9f9;
        border-left: 4px solid #4A90E2;
        color: black;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .answer-box {
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        background-color: #e3f2fd;
        border-left: 4px solid #42a5f5;
        color: black;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .divider-line {
        border-top: 1px solid #ddd;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .button-row {
        display: flex;
        justify-content: space-between;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>정보처리기사 실기 기반 챗봇</h1>", unsafe_allow_html=True)
st.write("기출문제에 대한 질문을 입력하여 정보를 확인해보세요.")
st.write("예시질문 1. 20년 1회 1번 문제에 대해 설명해줘")
st.write("예시질문 2. 20년 1회 1번 문제와 비슷한 문제를 만들어줘")

if "qna_history" not in st.session_state:
    st.session_state.qna_history = load_history_from_file()

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=0, file_label=""):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return [{"text": chunk, "source": file_label} for chunk in chunks]

def load_all_pdfs_in_directory(directory_path):
    all_chunks = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            text = extract_text_from_pdf(pdf_path)
            match = re.search(r"(\d{2})년(\d)회", filename)
            if match:
                year, round_num = match.groups()
                file_label = f"{year}년{round_num}회"
                chunks = split_text_into_chunks(text, file_label=file_label)
            else:
                file_label = filename
                chunks = split_text_into_chunks(text, file_label=file_label)
            all_chunks.extend(chunks)
    return all_chunks

def create_qa_chain_for_specific_chunks(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts([chunk["text"] for chunk in chunks], embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 11})
    llm = ChatOpenAI(model_name="gpt-4-turbo", max_tokens=512)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

directory_path = "./data"
all_chunks = load_all_pdfs_in_directory(directory_path)

def extract_year_round_from_question(question):
    match = re.search(r"(\d{2,4})년\s*(\d+)회", question)
    if match:
        year, round_num = match.groups()
        return year, round_num
    return None, None

user_question = st.text_input("질문을 입력하세요")

# 버튼 행
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("질문하기"):
        if user_question:
            year, round_num = extract_year_round_from_question(user_question)
            if year and round_num:
                file_keyword = f"{year}년{round_num}회"
                relevant_chunks = [chunk for chunk in all_chunks if file_keyword in chunk["source"]]

                if not relevant_chunks:
                    st.write("해당 연도와 회차에 맞는 정보가 없습니다.")
                else:
                    qa_chain = create_qa_chain_for_specific_chunks(relevant_chunks)
                    
            else:
                qa_chain = create_qa_chain_for_specific_chunks(all_chunks)

            generating_answer_container = st.empty()
            generating_answer_container.write("답변 생성 중...")

            result = qa_chain.invoke({"query": user_question})
            answer = result.get('result')

            generating_answer_container.empty()

            if answer:
                st.session_state.qna_history.insert(0, (user_question, answer))
            else:
                st.session_state.qna_history.insert(0, (user_question, "해당 정보는 존재하지 않습니다."))
            
            save_history_to_file(st.session_state.qna_history)

with col2:
    if st.button("채팅 내역 삭제"):
        st.session_state.qna_history = []
        save_history_to_file(st.session_state.qna_history)

st.subheader("이전 질문과 답변")

for i, (question, answer) in enumerate(st.session_state.qna_history, 1):
    st.markdown(f"<div class='question-box'><b>{i}. 질문</b>: {question}</div>", unsafe_allow_html=True)
    
    if i == 1:
        answer_container = st.empty()
        full_answer = ""
        for line in answer.split('\n'):
            full_answer += line + "\n"
            answer_container.markdown(f"<div class='answer-box'><b>답변</b>: {full_answer}</div>", unsafe_allow_html=True)
            time.sleep(0.1)
    else:
        st.markdown(f"<div class='answer-box'><b>답변</b>: {answer}</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='divider-line'></div>", unsafe_allow_html=True)
