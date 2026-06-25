import streamlit as st
from datetime import datetime
import pandas as pd

st.set_page_config(
    page_title="Keep Track",
    page_icon="🦍",
    layout="wide"
)

today = datetime.now()

# ===== TITLE =====
st.title("Keep Track, ặc ặc")

# ===== DAY COUNTER / DATE =====
col1, col2 = st.columns([1, 3])

with col1:
    st.info("**Day 3**")

with col2:
    st.caption(today.strftime("%B %d, %Y"))

# ===== INTRODUCTION =====
st.markdown("""
note bài giảng + ôn hàm
""")

# ===== CONTENTS =====
st.markdown("## Các hàm đã dùng")
functions_data = {
    "No.": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "Functions": [
        "st.set_page_config()",
        "st.title()",
        "st.header()",
        "st.subheader()",
        "st.markdown()",
        "st.caption()",
        "st.info()",
        "st.success()",
        "st.write()",
        "st.columns()",
        "st.code()",
        "st.image()",
        "st.audio()",
        "st.latex()"
    ],
    "Purpose": [
    "Cấu hình tiêu đề trang, biểu tượng trang và bố cục hiển thị.",
    "Tạo tiêu đề chính của một trang.",
    "Tạo tiêu đề cho một phần lớn trong nội dung.",
    "Tạo tiêu đề cho một mục con trong nội dung.",
    "Hiển thị văn bản có định dạng Markdown, danh sách bullet và công thức LaTeX.",
    "Hiển thị văn bản nhỏ hơn, nhạt hơn, ví dụ như ngày tháng.",
    "Hiển thị hộp thông báo dạng thông tin.",
    "Hiển thị hộp thông báo nổi bật theo kiểu success.",
    "Hiển thị văn bản hoặc nội dung đầu ra tổng quát.",
    "Chia trang thành nhiều cột.",
    "Hiển thị khối code hoặc văn bản dạng monospace.",
    "Hiển thị hình ảnh.",
    "Nhúng trình phát âm thanh ",
    "Hiển thị công thức LaTeX dưới dạng một khối riêng."
],
"VD": [
    'page_title="Keep Track", layout="wide"',
    'st.title("Keep Track")',
    'st.header("1. Tổng quan về Vector Database và Embedding")',
    'st.subheader("2.1. Flat Index")',
    'Được dùng cho phần lớn nội dung ghi chú bài học',
    'Được dùng để hiển thị ngày hiện tại',
    'Được dùng cho Day 3 và các ghi chú hướng dẫn',
    'Được dùng để nhấn mạnh các ý chính trong phần PQ',
    'Được dùng cho các đoạn văn bản ngắn hoặc phần footer',
    'Được dùng để đặt Day 3 và ngày hiện tại nằm cạnh nhau',
    'Được dùng cho các ví dụ PQ như [3, 51, 199, 18]',
    'Có thể dùng cho sơ đồ và hình minh họa',
    'Có thể dùng cho phần âm thanh giải thích bài học',
    'Có thể dùng cho các công thức trong distance metrics và loss functions'
]
}

df_functions = pd.DataFrame(functions_data)

st.dataframe(df_functions, use_container_width=True, hide_index=True)