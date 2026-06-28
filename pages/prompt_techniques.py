import streamlit as st
from streamlit_mermaid import st_mermaid
import pandas as pd

st.title("Prompt Techniques")

st.code("""
Prompt Engineering
│
├── 1. Basic Prompting: Có hay không có ví dụ?
│     ├── Zero-shot
│     └── Few-shot
│
├── 2. Reasoning Prompting: người dùng gợi ý cho mô hình phương thức suy luận
│     ├── Chain-of-Thought (CoT)
│     │      ├── Zero-shot CoT
│     │      └── Automatic CoT
│     │
│     ├── Self-Consistency
│     ├── Generated Knowledge
│     └── Tree of Thoughts
│
├── 3. Tool-assisted Reasoning: Cho phép mô hình tự suy luận và quyết định xem có gọi công cụ hỗ trợ không?
│     ├── PAL
│     └── ReAct
│
├── 4. Self-reflective Reasoning: Mô hình tự kiểm tra và cải thiệt sau khi có đáp án
│     └── Reflexion
│
├── 5. Prompt Optimization: mô hình tự sinh nhiều prompt, sau đó đánh giá prompt nào hoạt động tốt nhất
│     ├── Automatic Prompt Engineer
│     ├── Active-Prompt
│     ├── Directional Stimulus
│     └── Synthetic Prompting
│
└── 6. Multimodal Prompting: Nhận nhiều loại dữ liệu ngoài text rồi mới đưa ra câu trả lời
      └── Multimodal CoT
""")

st.subheader("🦍🦍🦍 Tiêu chí phân nhóm Prompt Engineering")

st.markdown("""
Các kỹ thuật Prompt Engineering chia thành **6 nhóm chính** căn cứ trên câu hỏi cần giải quyết khi làm việc với các mô hình LLMs.
""")

prompt_groups = {
    "Question": [
        "Có ví dụ trong promt hay không?",
        "Làm sao tôi giúp mô hình suy luận tốt hơn và đúng nhu cầu của tôi hơn?",
        "Khi nào vấn đề này cần dùng công cụ để giải quyết?",
        "Làm sao để mô hình tự kiểm tra và sửa giải pháp sau khi đưa ra giải pháp?",
        "Làm sao để mô hình chọn lựa được promt có thể cho ra kết quả tốt hơn?",
        "Mô hình nhận nhiều loại dữ liệu từ input để cho ra kết quả tốt hơn?"
    ],
    "Prompting Group": [
        "1. Basic Prompting",
        "2. Reasoning Prompting",
        "3. Tool-assisted Reasoning",
        "4. Self-reflective Reasoning",
        "5. Prompt Optimization",
        "6. Multimodal Prompting"
    ]
}

df_prompt_group = pd.DataFrame(prompt_groups)

st.dataframe(
    df_prompt_group,
    use_container_width=True,
    hide_index=True
)

st.subheader('Nhóm 1. Basic')

basic_group = {"Zero-shot": 
               ["Đưa ra chỉ thị cho mô hình mà không cung cấp ví dụ",
                "Khi nhiệm vụ đơn giản hoặc mô hình đã có đủ kiến thức"
                ],
                "Few-shot":
                ["Cung cấp một vài ví dụ (examples) trước khi đưa yêu cầu mới",
                 "Khi muốn mô hình học theo định dạng, phong cách hoặc quy tắc cụ thể"
                ]}

df_basic = pd.DataFrame(basic_group)
st.dataframe(df_basic, use_container_width=True, hide_index=True)

st.subheader("Nhóm 2. Reasoning Prompting")

reasoning_group = {
    "Chain-of-Thought (CoT)": [
        "Yêu cầu mô hình suy nghĩ từng bước trước khi đưa ra câu trả lời",
        "Dùng cho các bài toán cần lập luận hoặc tính toán"
    ],
    "Zero-shot CoT": [
        "Không ví dụ, chỉ thêm câu như 'Let's think step by step'",
        "Kích hoạt khả năng suy luận của mô hình"
    ],
    "Automatic CoT": [
        "Tự động sinh các chuỗi suy luận mẫu thay vì viết thủ công",
        "Giảm công sức tạo ví dụ CoT"
    ],
    "Self-Consistency": [
        "Sinh nhiều chuỗi suy luận và chọn đáp án xuất hiện nhiều nhất",
        "Tăng độ chính xác và tính ổn định"
    ],
    "Generated Knowledge": [
        "Sinh thêm kiến thức liên quan trước khi trả lời",
        "Giúp cải thiện chất lượng suy luận"
    ],
    "Tree of Thoughts": [
        "Khám phá nhiều hướng suy luận như một cây tìm kiếm",
        "Phù hợp với bài toán phức tạp cần nhiều phương án"
    ]
}

df_reasoning = pd.DataFrame(reasoning_group)
st.dataframe(df_reasoning, use_container_width=True, hide_index=True)

st.subheader("Nhóm 3. Tool-assisted Reasoning")

tool_group = {
    "PAL": [
        "Sinh chương trình (thường là Python) để giải bài toán",
        "Phù hợp với tính toán và lập trình"
    ],
    "ReAct": [
        "Kết hợp Reasoning và Action để sử dụng công cụ",
        "Phù hợp với Agent, Search và Tool Calling"
    ]
}

df_tool = pd.DataFrame(tool_group)
st.dataframe(df_tool, use_container_width=True, hide_index=True)

st.subheader("Nhóm 4. Self-reflective Reasoning")

reflection_group = {
    "Reflexion": [
        "Tự đánh giá và sửa câu trả lời sau khi hoàn thành",
        "Giúp giảm lỗi và cải thiện chất lượng đầu ra"
    ]
}

df_reflection = pd.DataFrame(reflection_group)
st.dataframe(df_reflection, use_container_width=True, hide_index=True)

st.subheader("Nhóm 5. Prompt Optimization")

optimization_group = {
    "Automatic Prompt Engineer": [
        "Tự động sinh nhiều prompt và chọn prompt tốt nhất",
        "Giảm việc thiết kế prompt thủ công"
    ],
    "Active-Prompt": [
        "Chọn các ví dụ few-shot hiệu quả nhất",
        "Cải thiện chất lượng prompt bằng example selection"
    ],
    "Directional Stimulus": [
        "Thêm gợi ý để định hướng mô hình",
        "Giúp mô hình tập trung vào thông tin quan trọng"
    ],
    "Synthetic Prompting": [
        "Tự sinh thêm prompt hoặc dữ liệu huấn luyện",
        "Tăng tính đa dạng và khả năng tổng quát hóa"
    ]
}

df_optimization = pd.DataFrame(optimization_group)
st.dataframe(df_optimization, use_container_width=True, hide_index=True)

st.subheader("Nhóm 6. Multimodal Prompting")

multimodal_group = {
    "Multimodal CoT": [
        "Kết hợp hình ảnh và văn bản trong quá trình suy luận",
        "Phù hợp với các tác vụ Vision-Language"
    ]
}

df_multimodal = pd.DataFrame(multimodal_group)
st.dataframe(df_multimodal, use_container_width=True, hide_index=True)

st.warning('''Một Prompt Template tối thiểu thường gồm 2 thành phần:
Instruction 
    +
Input ''')