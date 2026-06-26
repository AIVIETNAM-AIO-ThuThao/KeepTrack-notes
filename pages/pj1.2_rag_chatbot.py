import streamlit as st


st.title("PROJECT 1.2. RAG CHATBOT")

st.header("1. Giới thiệu chung và chuẩn bị môi trường")
st.markdown("""** Retrieval Augmented Generation (RAG)** là một kỹ thuật giúp truy vấn dữ liệu có liên quan từ tài liệu có sẵn, chúng ta tìm những đoạn văn bản liên
rồi đưa các đoạn đó kèm theo câu hỏi cho LLM. Nhờ vậy, LLM có thể trả lời chính xác cho câu hỏi từ người dùng.

RAG chatbot có: 
- Input: File tài liệu PDF cần hỏi đáp và một câu hỏi liên quan đến nội dung tài liệu
- Output: Câu trả lời dựa trên nội dung tài liệu
""")

st.markdown("### Chuẩn bị môi trường:")
st.markdown("##### **Ollama**")
st.markdown("""Ollama là một công cụ mã nguồn mở giúp tải và chạy các mô hình ngôn ngữ lớn (LLM) trên máy tính cá nhân. 
Ollama hoạt động như một "server nhỏ" chạy ngầm trên máy, các chương trình Python gọi vào qua API. Cần cài qua PowerShell hoặc download file .exe từ https://ollama.com/download""")
st.markdown("##### Thư viện")
st.code("pip install -q pypdf chromadb ollama       # install trong Terminal của VSCode", language="python")
st.code("""
        import pypdf        # Đọc nội dung file PDF
        import chromadb     # Lưu trữ và tìm kiếm vector (vector database)
        import ollama       # Giao tiếp với Ollama server để tạo embedding và sinh văn bản""",
        language="python")

st.header("2. Xây dựng chương trình RAG")
st.markdown("### pipeline RAG (5 bước):")
st.markdown("""**Đọc dữ liệu → Chunking → Embedding + Lưu vector vào database → Retrieve → Hỏi đáp**     
            """)
st.subheader("2.1. Đọc toàn bộ nội dung từ 1 file PDF")
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

st.markdown("#### Hàm chunking bằng Python thuần:")
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

st.subheader("2.3. Embedding và lưu vào Vector Database")

st.markdown('''**Embedding** là quá trình chuyển văn bản thành một dãy số (vector) sao cho các đoạn văn có ý nghĩa giống nhau sẽ có vector gần nhau trong không gian. Ví dụ: câu "YOLO là mô hình phát hiện vật thể" và "Object Detection với YOLO” sẽ có vector rất gần nhau, dù
dùng từ ngữ khác nhau. Hãy hình dung đơn giản: nếu mỗi đoạn văn bản là một điểm trên bản đồ, thì embedding giúp đặt các đoạn có nội dung giống nhau gần nhau trên bản đồ đó.

            ''')
st.markdown("### Hàm tạo embedding từ danh sách text")
st.code(r"""     
# Bước 1: Embedding - Gọi model embedding từ Ollama "bge-m3" để chuyển đổi text nhập vào thành vector số học
def embed(texts):
    return ollama.embed(model=, input=texts)["embeddings"]

'''Bước 2: Tạo vector database trong bộ nhớ và chromadb
- Client tạo kết nối đến database (ở đây là in-memory để chạy nhanh)
-Dữ liệu sẽ mất khi tắt chương trình. Nếu muốn lưu lâu dài, dùng chromadb.PersistentClient(path="./chroma_db").'''
client = chromadb.Client()
collection = client.get_or_create_collection("rag") # Tạo hoặc lấy collection có tên "rag" để lưu trữ dữ liệu


# Bước 3 INDEXING (đánh chỉ mục)
# Thêm tất cả chunks vào database
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,       # Lưu nội dung text gốc của từng chunk
    embeddings=embed(chunks),           # Tạo và lưu vector embeddings tương ứng cho mỗi chunk
)

print("Đã index:", collection.count(), "chunks"))  # In ra số lượng chunks đã được index thành công""",
language="python")   

st.subheader("2.4. Retrieve")
st.markdown('''Khi có câu hỏi, chúng ta cần tìm những chunk có nội dung liên quan nhất. 
            ChromaDB thực hiện việc này bằng cách so sánh vector của câu hỏi với tất cả vector trong database''')
st.code(r"""
    def retrieve(query, k=4):       #Tìm k đoạn văn bản liên quan nhất với câu hỏi
    
    # Gọi phương thức query của collection trong ChromaDB
    # Vector hóa câu hỏi của người dùng và so sánh với các vector chunks trong database
        res = collection.query(        
            query_embeddings=embed([query]),
            n_results=k      # Số lượng kết quả muốn trả về (mặc định là 4)
    )
        return res["documents"][0]      # Trả về kết quả đầu tiên trong danh sách các chunks đã tìm được
# Thử tìm kiếm
    QUERY = "YOLOv10 dùng để làm gì?"
    for doc in retrieve(QUERY):
        print(doc[:200])
        print("-" * 40)""", 
    language="python")
st.markdown('''####
        - K càng lớn: context càng nhiều nhưng có thể có nhiễu
        - K càng nhỏ: context ngắn nhưng có thể thiếu thông tin
        - Thường chọn k=3-5 để cân bằng giữa chất lượng và hiệu quả''')
