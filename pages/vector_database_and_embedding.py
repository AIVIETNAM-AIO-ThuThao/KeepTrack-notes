import streamlit as st
import pandas as pd


st.title("Vector Database and Embedding")

st.header("1. Tổng quan về Vector Database và Embedding")

st.markdown("""
**Vector Database** (hay còn gọi là *vector store* hoặc *vector search engine*) là một loại cơ sở dữ liệu được thiết kế chuyên biệt để **lưu trữ, chỉ mục và truy vấn các vector embeddings** — tức các biểu diễn số của dữ liệu phức tạp như **hình ảnh, văn bản, âm thanh, sensor data,...**. Vector Database đặc biệt phù hợp với dữ liệu **đa chiều** (*high-dimensional data*), trong đó mỗi đối tượng được biểu diễn bằng một vector có thể gồm **hàng trăm đến hàng chục nghìn chiều**.
""")

st.markdown("""
**Embedding** là quá trình chuyển đổi dữ liệu thô (văn bản, hình ảnh, âm thanh, video,...) thành một **vector số nhiều chiều** (*multi-dimensional vector*), trong đó mỗi chiều đại diện cho một đặc trưng (*feature*) ngữ nghĩa hoặc hình thái của dữ liệu.

Việc biểu diễn dữ liệu dưới dạng vector giúp các hệ thống AI và Vector Database có thể:
- tính toán **độ tương đồng** giữa các đối tượng,
- thực hiện **truy vấn nearest neighbor**,
- xử lý dữ liệu phức tạp mà không cần làm việc trực tiếp với ngôn ngữ tự nhiên, hình ảnh thô hay tín hiệu âm thanh.
""")


st.image('assets/vectordatabase.JPG', caption='Vector Database', width=900)
st.image('assets/embedding.JPG', caption='Embedding', use_container_width=True)
st.header("2. So sánh công cụ Vector Database Tools")
# tạo dict
comparation = {     
    "Tool": ['Faiss', 'Milvus', 'Weaviate', 'Weaviate', 'Weaviate'],
    "Đặc điểm": [
        "Library C++/Python, không có persistence",
        "Distributed, persistent, REST API",
        "Semantic search, GraphQL API",
        "SaaS, dễ deploy, auto-scale",
        "Extension cho PostgreSQL"
        ],
    "Iu điểm": [
    "Hiệu năng rất cao, nhiều thuật toán ANN",
    "Production-ready, hỗ trợ nhiều metric",
    "Tích hợp mô hình ngôn ngữ, dễ dùng",
    "Không cần quản lý hạ tầng",
    "Dễ tích hợp với RDBMS",
],
"Khuyết điểm": [
    'Không lưu trữ dữ liệu lâu dài',
    'Cần vận hành hạ tầng riêng',
    'Hạn chế backend tùy biến',
    'Phụ thuộc vào nhà cung cấp',
    'Hiệu năng thấp hơn chuyên dụng',
    
]
}
# chuyển dict thành dataframe
df_comparation = pd.DataFrame(comparation)
# hàm của streamlit giúp hiển thị dataframe trên website
st.dataframe(df_comparation, use_container_width=True, hide_index=True)

# 2. INDEXING

st.header("3. Indexing trong Vector Database")

st.markdown("""
Sau khi dữ liệu đã được chuyển đổi thành vector embedding, bước tiếp theo là **indexing** — tức xây dựng **chỉ mục** để tối ưu hoá khả năng truy vấn và tìm kiếm.

Nếu không có chỉ mục, hệ thống thường phải so sánh vector truy vấn với toàn bộ vector trong cơ sở dữ liệu**, điều này trở nên rất tốn kém khi số lượng vector lớn.  
Do đó, các kỹ thuật indexing được sử dụng để:
- giảm thời gian tìm kiếm,
- tiết kiệm tài nguyên tính toán,
- hỗ trợ Approximate Nearest Neighbor Search (ANN) trên tập dữ liệu lớn.
""")

st.markdown("## Các loại index")


# 3.1 FLAT INDEX
st.subheader("3.1. Flat Index")

st.markdown("""
**Flat Index** là một tên gọi khác của tìm kiếm brute-force. Toàn bộ các vector được lưu trữ trong một cấu trúc chỉ mục duy nhất mà **không có bất kỳ tổ chức phân cấp nào**. Khi có một vector truy vấn, hệ thống sẽ tính khoảng cách từ vector đó tới **tất cả các vector trong dataset**, sau đó chọn ra vector gần nhất hoặc top-k vector gần nhất.
""")

st.markdown("### Đặc điểm:")
st.markdown("""
- **Chính xác tuyệt đối** vì so sánh với toàn bộ vector trong dataset.
- **Không cần build index phức tạp**.
- **Chi phí tính toán cao** nếu dataset lớn, đặc biệt khi số lượng vector lên tới hàng triệu hoặc hơn.
""")
st.image('assets/k flat.JPG', caption='Flat Index', use_container_width=True)
st.markdown("### Minh hoạ")
st.markdown("""
Nếu cơ sở dữ liệu có $N$ vector và mỗi vector có $d$ chiều, thì với một truy vấn $q$, Flat Index sẽ tính khoảng cách:
$$
d(q, x_i), \quad i = 1, 2, \dots, N
$$

với mọi vector $x_i$ trong cơ sở dữ liệu.
""")


# 3.2 IVF

st.subheader("3.2. Inverted File Index (IVF)")

st.markdown("""
**IVF (Inverted File Index)** là một kỹ thuật lập chỉ mục đơn giản và trực quan, thường được dùng trong các hệ thống truy xuất, và có thể được điều chỉnh cho cơ sở dữ liệu vector để thực hiện **Approximate Nearest Neighbor Search**. Ý tưởng chính của IVF là **không tìm kiếm trên toàn bộ dataset**, mà trước tiên chia dữ liệu thành nhiều **cluster**, sau đó chỉ tìm trong một số cluster phù hợp nhất với vector truy vấn.
""")

st.markdown("### Đặc điểm:")
st.markdown("""
- Sử dụng các thuật toán **phân cụm** để chia tất cả các vector trong dataset thành các vùng khác nhau (*clusters*).
- Mỗi cluster có một **trọng tâm** (centroid) tương ứng.
- Mỗi vector chỉ được gán vào **một cluster**, cụ thể là cluster có centroid gần nhất với vector đó.
- Mỗi centroid duy trì thông tin về các vector thuộc về phân vùng của nó.
""")
st.image('assets/cluster.JPG', caption='IVF Indexing', use_container_width=True)
st.markdown("### Cách truy vấn với IVF")
st.markdown("""
Khi có một vector truy vấn:

1. Hệ thống **không so sánh với toàn bộ vector trong database**.
2. Thay vào đó, nó tìm **centroid gần nhất** với vector truy vấn.
3. Sau đó chỉ tìm kiếm các vector nằm trong cluster tương ứng với centroid đó (hoặc một vài cluster gần nhất nếu cần).

Nhờ vậy, không gian tìm kiếm được thu hẹp đáng kể, giúp truy vấn nhanh hơn so với Flat Index.
""")

st.markdown("### Tóm tắt trực quan")
st.markdown("""
- **Flat Index**: tìm trên toàn bộ dataset  
- **IVF**: tìm cluster gần nhất trước, rồi chỉ tìm trong cluster đó
""")




# 3.3 PQ

st.subheader("3.3. Product Quantization (PQ)")

st.markdown("""
**Product Quantization (PQ)** là một kỹ thuật **nén vector** và **tìm kiếm xấp xỉ** (*Approximate Nearest Neighbor Search*).  
Mục tiêu của PQ là giảm đáng kể chi phí lưu trữ và tăng tốc truy vấn bằng cách thay thế vector gốc bằng một **mã nén ngắn hơn**.
""")

st.markdown("---")
st.markdown("## Cách hoạt động của Product Quantization")

# STEP 1
st.markdown("### Bước 1: Chia nhỏ vector (Phân vùng)")

st.markdown("""
Giả sử mỗi vector trong database có chiều dài $D = 128$.  
PQ sẽ chia vector này thành $m$ nhóm con (*sub-vectors*). Ví dụ, nếu chọn $m = 4$, thì vector sẽ được chia thành 4 phần.

Ví dụ minh hoạ:
""")

st.code(
"""Vector gốc:
[v1, v2, v3, v4 | v5, v6, v7, v8 | v9, v10, v11, v12 | v13, v14, v15, v16]""",
language="text"
)

st.markdown("""
Như vậy:
- vector gốc được chia thành **4 không gian con** (*subspaces*),
- mỗi không gian con chỉ chứa một phần nhỏ số chiều ban đầu.
""")

# STEP 2
st.markdown("### Bước 2: Gán mã (Lượng tử hóa từng phần)")

st.markdown("""
Trong mỗi không gian con, ta chạy thuật toán **K-means** để học ra $k$ cụm (*centroids*).  
Thông thường, $k = 256$ để mỗi centroid có thể được biểu diễn bằng **1 byte**.

Ví dụ:
- Không gian con thứ 1: học ra 256 centroid, ký hiệu từ $C_{1,1}$ đến $C_{1,256}$
- Không gian con thứ 2: học ra 256 centroid, ký hiệu từ $C_{2,1}$ đến $C_{2,256}$
- ...
- Tương tự cho tất cả các không gian con còn lại
""")

st.markdown("""
Sau đó, **mỗi sub-vector** sẽ được thay thế bằng **ID của centroid gần nhất**.

Nói cách khác, thay vì lưu toàn bộ giá trị số thực của sub-vector, ta chỉ lưu **mã số của cụm** mà nó thuộc về.
""")

# STEP 3
st.markdown("### Bước 3: Lưu trữ dưới dạng mã (Codebook Representation)")

st.markdown("""
Thay vì lưu vector gốc gồm rất nhiều số thực (*float values*), PQ chỉ lưu một chuỗi **mã số nguyên ngắn**.

Ví dụ:
- Vector A sau khi nén: `[1, 53, 200, 15]`
- Vector B sau khi nén: `[3, 51, 199, 18]`
""")

st.markdown("""
Điều này giúp giảm mạnh bộ nhớ lưu trữ.  
Ví dụ:
- vector gốc có thể chiếm khoảng **512 bytes**
- sau khi nén bằng PQ, chỉ còn **4 bytes** nếu dùng 4 mã, mỗi mã 1 byte
""")

st.success("PQ giúp đánh đổi một phần độ chính xác để đạt được khả năng lưu trữ gọn hơn và truy vấn nhanh hơn trên tập dữ liệu lớn.")

# QUERY WITH PQ
st.markdown("---")
st.markdown("## Tìm kiếm (Query) với PQ")

st.markdown("""
PQ sử dụng kỹ thuật **ADC (Asymmetric Distance Computation)** để ước lượng khoảng cách giữa **vector truy vấn** và các vector đã được mã hoá trong cơ sở dữ liệu.
""")

st.markdown("### Quy trình truy vấn với ADC")

st.markdown("""
**Bước 1: Chia vector query thành 4 nhóm con giống như trên**  

Vector truy vấn cũng được chia thành 4 nhóm con giống như cách đã chia vector trong database.
""")

st.markdown("""
**Bước 2: Tính sẵn khoảng cách từ query đến từng centroid (256 cụm) trong mỗi không gian con**  

Trong mỗi không gian con, ta tính khoảng cách từ sub-vector của query đến toàn bộ các centroid đã học bằng K-means.
""")

st.markdown("""
**Bước 3: Lưu các khoảng cách này vào một bảng tra cứu (lookup table)**  

Bảng tra cứu có kích thước:

$$
m \\times k
$$

Ví dụ, nếu:
- $m = 4$
- $k = 256$

thì bảng tra cứu sẽ có kích thước:

$$
4 \\times 256
$$
""")

st.markdown("""
**Bước 4: So sánh với một vector bất kỳ trong database**  

Giả sử một vector trong database đã được nén thành mã:
""")

st.code("[3, 51, 199, 18]", language="text")

st.markdown("""
Khi đó, để ước lượng khoảng cách giữa **query vector** và vector này, ta chỉ cần:

- lấy khoảng cách đã tính sẵn của query với **centroid số 3** trong **nhóm 1**,
- cộng với khoảng cách đã tính sẵn của query với **centroid số 51** trong **nhóm 2**,
- cộng với khoảng cách đã tính sẵn của query với **centroid số 199** trong **nhóm 3**,
- cộng với khoảng cách đã tính sẵn của query với **centroid số 18** trong **nhóm 4**.

Tức là khoảng cách xấp xỉ được tính như sau:

$$
d(q, x) \\approx d(q_1, C_{1,3}) + d(q_2, C_{2,51}) + d(q_3, C_{3,199}) + d(q_4, C_{4,18})
$$

Trong đó:
- $q_1, q_2, q_3, q_4$ là 4 phần của query vector,
- $C_{1,3}$ là centroid số 3 của không gian con thứ nhất,
- $C_{2,51}$ là centroid số 51 của không gian con thứ hai,
- $C_{3,199}$ là centroid số 199 của không gian con thứ ba,
- $C_{4,18}$ là centroid số 18 của không gian con thứ tư.

Tổng các giá trị trên chính là **khoảng cách xấp xỉ** giữa query vector và vector trong cơ sở dữ liệu.
""")
