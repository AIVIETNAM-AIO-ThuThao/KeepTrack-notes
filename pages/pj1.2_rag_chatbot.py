import streamlit as st


st.title("PROJECT 1.2. RAG CHATBOT")

st.header("1. Giới thiệu chung và chuẩn bị môi trường")
st.markdown("""Retrieval Augmented Generation (RAG) là một kỹ thuật giúp truy vấn dữ liệu có liên quan từ tài liệu có sẵn, chúng ta tìm những đoạn văn bản liên
rồi đưa các đoạn đó kèm theo câu hỏi cho LLM. Nhờ vậy, LLM có thể trả lời chính xác cho câu hỏi từ người dùng.

RAG chatbot có: 
- Input: File tài liệu PDF cần hỏi đáp và một câu hỏi liên quan đến nội dung tài liệu
- Output: Câu trả lời dựa trên nội dung tài liệu
""")

st.markdown("## Chuẩn bị môi trường")
st.markdown("### **Ollama**")
st.markdown("""Ollama là một công cụ mã nguồn mở giúp tải và chạy các mô hình ngôn ngữ lớn (LLM) trên máy tính cá nhân. 
Ollama hoạt động như một "server nhỏ" chạy ngầm trên máy, các chương trình Python gọi vào qua API. Cần cài qua PowerShell hoặc download file .exe từ https://ollama.com/download""")
st.markdown("### Thư viện")
st.code("""
        import pypdf        # Đọc nội dung file PDF
        import chromadb     # Lưu trữ và tìm kiếm vector (vector database)
        import ollama       # Giao tiếp với Ollama server để tạo embedding và sinh văn bản""",
        language="python")

st.header("2. Xây dựng chương trình RAG")
st.markdown("## pipeline RAG theo 5 bước")