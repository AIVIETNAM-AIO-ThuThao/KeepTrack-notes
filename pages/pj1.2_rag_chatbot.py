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
st.code("pip install -q pypdf chromadb ollama       # install trong Terminal của VSCode", language="python")
st.code("""
        import pypdf        # Đọc nội dung file PDF
        import chromadb     # Lưu trữ và tìm kiếm vector (vector database)
        import ollama       # Giao tiếp với Ollama server để tạo embedding và sinh văn bản""",
        language="python")

st.header("2. Xây dựng chương trình RAG")
st.markdown("### pipeline RAG (5 bước):")
st.markdown("""**Đọc PDF → Cắt nhỏ → Embedding + Lưu vector → Tìm kiếm → Hỏi đáp**     
            """)
st.subheader("2.1. Đọc toàn bộ nội dung từ file PDF")
st.code(r"""
# Đọc file PDF
reader = pypdf.PdfReader("./YOLOv10_Tutorials.pdf")

# Ghép nội dung tất cả các trang thành 1 chuỗi text
full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

print("Số trang:", len(reader.pages))
print("Tổng ký tự:", len(full_text)))""",
 language="python")

st.subheader("2.2. Chunking")
st.markdown("""
            Tại một thời điểm, model chỉ chấp nhận một tổng lượng văn bản (tính bằng token). Nếu nhồi quá nhiều văn bản vượt mức giới hạn này, mô hình sẽ báo lỗi (vượt số token cho phép) hoặc bị cắt bớt khiến model "quên" mất thông tin ở đoạn đầu. 
            **Chunking** là quá trình cắt nhỏ văn bản thành các đoạn ngắn để không gây quá tải cho model phân tích.
            """)
st.image("assets/chunking.JPG", caption="Minh họa quá trình chunking", width=900)

st.markdown("## Hàm chunking bằng Python thuần:")
st.markdown("Cắt text thành các đoạn nhỏ có độ dài tối đa 'size' = 1000 ký tự, với 'overlap' ký tự trùng lặp giữa 2 đoạn liên tiếp là 200")
st.code(r"""
def chunk_text(text, size=1000, overlap=200):

    # Xử lý văn bản đầu vào: tách theo dòng mới, loại bỏ khoảng trắng thừa và bỏ qua dòng trống
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    
    # Khởi tạo danh sách chunks và biến cur để xây dựng chunk hiện tại
    chunks, cur = [], ""
    
    # Duyệt qua từng đoạn văn (paragraph), không cắt giữa chừng đoạn văn
    for p in paras:
        # Kiểm tra nếu thêm đoạn văn mới vào chunk hiện tại không làm vượt quá giới hạn size
        # (+1 cho ký tự newline \n)
        if len(cur) + len(p) + 1 <= size:
            # Thêm đoạn văn vào chunk hiện tại, kèm theo dấu xuống dòng
        
            cur += p + "\n"
        else:   # Nếu chunk hiện tại đã đầy, lưu nó vào danh sách chunks (nếu có nội dung)
            if cur:
                chunks.append(cur.strip())
            
            ''' Tạo chunk mới với phần overlap (trùng lặp) từ chunk trước
            - Giúp duy trì ngữ cảnh xuyên suốt giữa các chunk
            - Tránh mất mát thông tin quan trọng ở biên giữa các chunk
           '''
            cur = (cur[-overlap:] + p + "\n") if overlap else (p + "\n")
    
    # Xử lý chunk cuối cùng sau khi duyệt hết các đoạn văn
    if cur.strip():
        chunks.append(cur.strip())
    
    return chunks       # Trả về danh sách các chunk đã được cắt

# Cắt văn bản full_text thành các đoạn nhỏ có thể truy xuất được
chunks = chunk_text(full_text)

# In ra số lượng chunks đã tạo để kiểm tra kích thước dữ liệu, đánh giá hiệu quả chia chunk
print("Số chunks:", len(chunks))

# Xem trước 300 ký tự đầu của chunk đầu tiên để kiểm tra chất lượng
print(chunks[0][:300])
""", 
language="python")

st.subheader("🦧 Chọn size của chunk và chunk_overlap phù hợp")
st.markdown('''
            - chunk_size quá nhỏ (ví dụ: 200) sẽ làm mất ngữ cảnh, câu trả lời thiếu thông tin.
            - chunk_size quá lớn (ví dụ: 5000) sẽ lẫn nhiều thông tin nhiễu. 
            - Giá trị 1000 là một điểm khởi đầu tốt cho hầu hết các trường hợp. 
            - size của chunk_overlap nên bằng 10-20% của chunk''')