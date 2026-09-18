# -*- coding: utf-8 -*-
"""
HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH (BCTC)
Ứng dụng Web phát hiện gian lận dựa trên mô hình Hồi quy Logistic & Mô hình Beneish M-Score
Tác giả: Chuyên gia Phân tích Tài chính & Kỹ sư Web App
"""

import io
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import streamlit as st

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH TRANG WEB & GIAO DIỆN
# ==============================================================================
st.set_page_config(
    page_title="Dự Báo Gian Lận BCTC | Financial Fraud Detector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tùy biến CSS giao diện hiện đại, chuyên nghiệp
st.markdown(
    """
    <style>
    /* Font & container styling */
    .main {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-card-danger {
        background-color: #fef2f2;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 5px solid #ef4444;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-card-success {
        background-color: #f0fdf4;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 5px solid #22c55e;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .badge-safe {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-danger {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. HẰNG SỐ & ĐẶC TRƯNG M-SCORE
# ==============================================================================
EXPECTED_FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]
TARGET = "FRAUD_FLAG"
DEFAULT_CSV = "MScore_data.csv"

# Thông tin chi tiết về từng chỉ số Beneish
FEATURE_INFO = {
    "DSRI": {
        "name": "Days Sales in Receivables Index",
        "vn_name": "Số ngày phải thu khách hàng",
        "default": 1.05,
        "step": 0.05,
        "format": "%.3f",
        "red_flag": 1.30,
        "flag_direction": ">",
        "desc": "Tỷ lệ số ngày thu tiền khách hàng kỳ này so với kỳ trước. DSRI > 1.30 báo hiệu nguy cơ ghi nhận doanh thu ảo hoặc nới lỏng tín dụng để thổi phồng doanh số.",
    },
    "GMI": {
        "name": "Gross Margin Index",
        "vn_name": "Tỷ suất lãi gộp suy giảm",
        "default": 1.02,
        "step": 0.05,
        "format": "%.3f",
        "red_flag": 1.20,
        "flag_direction": ">",
        "desc": "Tỷ lệ biên lãi gộp năm trước so với năm nay. GMI > 1.20 cho thấy biên lãi gộp đang xấu đi đáng kể, tạo áp lực gian lận để duy trì kỳ vọng thị trường.",
    },
    "AQI": {
        "name": "Asset Quality Index",
        "vn_name": "Chất lượng tài sản",
        "default": 1.00,
        "step": 0.05,
        "format": "%.3f",
        "red_flag": 1.25,
        "flag_direction": ">",
        "desc": "Tỷ lệ tài sản phi hiện hữu (ngoài TSCĐ, tiền, hàng tồn kho). AQI > 1.25 phản ánh việc vốn hóa chi phí bất thường vào tài sản để trì hoãn ghi nhận lỗ.",
    },
    "SGI": {
        "name": "Sales Growth Index",
        "vn_name": "Tăng trưởng doanh thu",
        "default": 1.10,
        "step": 0.05,
        "format": "%.3f",
        "red_flag": 1.30,
        "flag_direction": ">",
        "desc": "Tỷ lệ doanh thu năm nay so với năm trước. Tăng trưởng nóng (SGI > 1.30) thường đi kèm rủi ro bóp méo số liệu kế toán khi đà tăng chậm lại.",
    },
    "DEPI": {
        "name": "Depreciation Index",
        "vn_name": "Tỷ lệ khấu hao tài sản",
        "default": 1.00,
        "step": 0.05,
        "format": "%.3f",
        "red_flag": 1.15,
        "flag_direction": ">",
        "desc": "Tỷ lệ khấu hao năm trước so với năm nay. DEPI > 1.15 cho thấy doanh nghiệp đang kéo dài thời gian khấu hao hoặc áp dụng phương pháp khấu hao chậm hơn để tăng lợi nhuận.",
    },
    "SGAI": {
        "name": "Sales, General & Admin Expenses Index",
        "vn_name": "Chi phí bán hàng & QLDN",
        "default": 1.00,
        "step": 0.05,
        "format": "%.3f",
        "red_flag": 1.20,
        "flag_direction": ">",
        "desc": "Tỷ lệ chi phí SG&A trên doanh thu kỳ này so với kỳ trước. SGAI > 1.20 phản ánh hiệu quả quản lý chi phí suy giảm.",
    },
    "TATA": {
        "name": "Total Accruals to Total Assets",
        "vn_name": "Biến dồn tích trên tổng tài sản",
        "default": 0.03,
        "step": 0.01,
        "format": "%.3f",
        "red_flag": 0.08,
        "flag_direction": ">",
        "desc": "Chênh lệch giữa lợi nhuận kế toán và dòng tiền hoạt động thuần chia cho tổng tài sản. TATA càng dương lớn (> 0.08) chứng tỏ lợi nhuận không có tiền mặt bảo đảm.",
    },
    "LVGI": {
        "name": "Leverage Index",
        "vn_name": "Đòn bẩy tài chính",
        "default": 1.05,
        "step": 0.05,
        "format": "%.3f",
        "red_flag": 1.25,
        "flag_direction": ">",
        "desc": "Tỷ lệ tổng nợ trên tổng tài sản kỳ này so với kỳ trước. LVGI > 1.25 thể hiện áp lực đòn bẩy tài chính gia tăng, dễ dẫn đến động cơ che giấu nợ hoặc rủi ro vỡ nợ.",
    },
}

# ==============================================================================
# 3. HÀM TÍNH BENEISH M-SCORE CHUẨN (1999)
# ==============================================================================
def calculate_beneish_mscore(dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi):
    """
    Công thức Beneish M-Score 8 biến (Beneish, 1999):
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.037*TATA + 0.0327*LVGI
    Ngưỡng phân loại:
    - Nếu M > -1.78: Khả năng cao doanh nghiệp can thiệp/bóp méo BCTC (Gian lận)
    - Nếu M <= -1.78: Nguy cơ gian lận thấp (An toàn)
    """
    m_score = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.037 * tata
        + 0.0327 * lvgi
    )
    is_manipulator = m_score > -1.78
    return m_score, is_manipulator

# ==============================================================================
# 4. HUẤN LUYỆN VÀ CACHE MÔ HÌNH HỒI QUY LOGISTIC
# ==============================================================================
@st.cache_resource(show_spinner="Đang tải dữ liệu và huấn luyện mô hình Logistic Regression...")
def train_logistic_model(data_path=DEFAULT_CSV):
    """Đọc dữ liệu CSV và huấn luyện pipeline Logistic Regression chuẩn hoá."""
    if not Path(data_path).exists():
        return None, None, None, None, None, None, None, None

    df = pd.read_csv(data_path)
    df.columns = df.columns.astype(str).str.strip().str.upper()

    if TARGET not in df.columns:
        raise ValueError(f"Không tìm thấy cột mục tiêu '{TARGET}' trong dữ liệu.")

    features = [c for c in EXPECTED_FEATURES if c in df.columns]
    for c in features + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Đánh giá trên tập kiểm tra test set
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Trích xuất hệ số về thang dữ liệu gốc
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    beta_std = model.coef_[0]
    beta_raw = beta_std / scaler.scale_
    intercept_raw = float(
        model.intercept_[0] - np.sum(beta_std * scaler.mean_ / scaler.scale_)
    )

    coef_df = pd.DataFrame(
        {
            "Biến số": features,
            "Tên tiếng Việt": [FEATURE_INFO[f]["vn_name"] for f in features],
            "Hệ số gốc (Beta)": beta_raw,
            "Hệ số chuẩn hóa": beta_std,
            "Odds Ratio (1 đv)": np.exp(beta_raw),
        }
    )

    return pipeline, X_train, X_test, y_train, y_test, y_prob, coef_df, intercept_raw

# ==============================================================================
# 5. SIDEBAR: THÔNG TIN VÀ CÀI ĐẶT
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/accounting.png", width=70)
    st.title("Fraud Radar BCTC")
    st.markdown("**Hệ thống chuyên gia phát hiện rủi ro gian lận Báo cáo Tài chính**")
    st.markdown("---")

    st.subheader("⚙️ Cấu Hình Mô Hình")
    threshold = st.slider(
        "Ngưỡng quyết định (Decision Threshold):",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Nếu Xác suất gian lận >= Ngưỡng này, mô hình sẽ dán cờ Cảnh báo Gian lận (1). Giảm ngưỡng giúp phát hiện triệt để hơn nhưng có thể tăng cảnh báo nhầm.",
    )

    st.markdown(
        f"""
        <div style='background-color:#e0f2fe; padding:10px; border-radius:8px; font-size:0.85rem; color:#0369a1;'>
        <b>Ngưỡng hiện tại: {threshold:.2f}</b><br>
        • P &ge; {threshold:.2f}: Nguy cơ cao (Gian lận)<br>
        • P &lt; {threshold:.2f}: An toàn / Rủi ro thấp
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("📌 Phương Pháp Đánh Giá")
    st.markdown(
        """
        1. **Mô hình Hồi quy Logistic (Machine Learning):**
           - Huấn luyện trên mẫu dữ liệu thực tế.
           - Tính toán xác suất gian lận $P(FRAUD=1)$.
        2. **Mô hình Beneish M-Score Chuẩn (1999):**
           - Tiêu chuẩn toàn cầu trong kiểm toán.
           - Ngưỡng phân định: $M > -1.78$.
        """
    )
    st.markdown("---")
    st.caption("Phiên bản 1.0 | Sẵn sàng triển khai trên Streamlit Cloud")

# ==============================================================================
# 6. KHỞI TẠO MÔ HÌNH
# ==============================================================================
pipeline, X_train, X_test, y_train, y_test, y_prob, coef_df, intercept_raw = train_logistic_model(DEFAULT_CSV)

if pipeline is None:
    st.error(
        f"⚠️ Không tìm thấy tệp dữ liệu mặc định '{DEFAULT_CSV}'. Vui lòng tải lên tệp dữ liệu huấn luyện hoặc đảm bảo '{DEFAULT_CSV}' nằm cùng thư mục với 'app.py'."
    )
    st.stop()

# ==============================================================================
# 7. TIÊU ĐỀ TRANG CHÍNH & TABS
# ==============================================================================
st.title("⚖️ Hệ Thống Dự Báo & Giám Sát Gian Lận BCTC")
st.markdown(
    "Ứng dụng kết hợp giữa **Khoa học Dữ liệu (Machine Learning)** và **Kế toán Pháp y (Forensic Accounting)** để rà soát, phát hiện sớm các hành vi thao túng, bóp méo số liệu Báo cáo tài chính doanh nghiệp."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔮 Dự Báo Doanh Nghiệp Đơn Lẻ",
        "📁 Dự Báo Hàng Loạt (File CSV/Excel)",
        "📊 Hiệu Năng & Hệ Số Mô Hình",
        "📚 Cẩm Nang 8 Chỉ Số Beneish",
    ]
)

# ==============================================================================
# TAB 1: DỰ BÁO DOANH NGHIỆP ĐƠN LẺ
# ==============================================================================
with tab1:
    st.subheader("Nhập Thông Số Tài Chính Để Phân Tích")
    st.markdown("Chọn kịch bản mẫu bên dưới để xem nhanh kết quả hoặc tự nhập số liệu thực tế:")

    # Nút chọn mẫu nhanh
    col_demo1, col_demo2, col_demo3 = st.columns([1, 1, 2])
    
    # State để quản lý giá trị nhập
    if "preset" not in st.session_state:
        st.session_state.preset = "normal"

    with col_demo1:
        if st.button("🟢 Mẫu: Doanh nghiệp An toàn", use_container_width=True):
            st.session_state.preset = "safe"
    with col_demo2:
        if st.button("🔴 Mẫu: Doanh nghiệp Rủi ro cao", use_container_width=True):
            st.session_state.preset = "risk"

    # Giá trị theo kịch bản
    if st.session_state.preset == "safe":
        vals = {"DSRI": 0.85, "GMI": 0.95, "AQI": 0.82, "SGI": 1.05, "DEPI": 0.98, "SGAI": 0.92, "TATA": 0.02, "LVGI": 0.95}
    elif st.session_state.preset == "risk":
        vals = {"DSRI": 1.65, "GMI": 1.45, "AQI": 1.38, "SGI": 1.55, "DEPI": 1.25, "SGAI": 1.22, "TATA": 0.16, "LVGI": 1.35}
    else:
        vals = {k: FEATURE_INFO[k]["default"] for k in EXPECTED_FEATURES}

    # Form nhập 8 chỉ số chia làm 4 cột
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Nhóm Doanh thu & Lợi nhuận**")
        dsri_val = st.number_input(
            "DSRI (Số ngày phải thu):",
            value=float(vals["DSRI"]),
            step=0.05,
            format="%.3f",
            help=FEATURE_INFO["DSRI"]["desc"],
        )
        gmi_val = st.number_input(
            "GMI (Tỷ suất lãi gộp):",
            value=float(vals["GMI"]),
            step=0.05,
            format="%.3f",
            help=FEATURE_INFO["GMI"]["desc"],
        )

    with col2:
        st.markdown("**Nhóm Chất lượng Tài sản**")
        aqi_val = st.number_input(
            "AQI (Chất lượng tài sản):",
            value=float(vals["AQI"]),
            step=0.05,
            format="%.3f",
            help=FEATURE_INFO["AQI"]["desc"],
        )
        sgi_val = st.number_input(
            "SGI (Tăng trưởng DT):",
            value=float(vals["SGI"]),
            step=0.05,
            format="%.3f",
            help=FEATURE_INFO["SGI"]["desc"],
        )

    with col3:
        st.markdown("**Nhóm Chi phí & Khấu hao**")
        depi_val = st.number_input(
            "DEPI (Tỷ lệ khấu hao):",
            value=float(vals["DEPI"]),
            step=0.05,
            format="%.3f",
            help=FEATURE_INFO["DEPI"]["desc"],
        )
        sgai_val = st.number_input(
            "SGAI (Chi phí SG&A):",
            value=float(vals["SGAI"]),
            step=0.05,
            format="%.3f",
            help=FEATURE_INFO["SGAI"]["desc"],
        )

    with col4:
        st.markdown("**Nhóm Dồn tích & Đòn bẩy**")
        tata_val = st.number_input(
            "TATA (Biến dồn tích):",
            value=float(vals["TATA"]),
            step=0.01,
            format="%.3f",
            help=FEATURE_INFO["TATA"]["desc"],
        )
        lvgi_val = st.number_input(
            "LVGI (Đòn bẩy nợ):",
            value=float(vals["LVGI"]),
            step=0.05,
            format="%.3f",
            help=FEATURE_INFO["LVGI"]["desc"],
        )

    # Đưa vào DataFrame dự báo
    input_data = pd.DataFrame(
        [[dsri_val, gmi_val, aqi_val, sgi_val, depi_val, sgai_val, tata_val, lvgi_val]],
        columns=EXPECTED_FEATURES,
    )

    # 1. Dự báo từ mô hình Logistic
    prob_fraud = pipeline.predict_proba(input_data)[0, 1]
    is_logistic_fraud = prob_fraud >= threshold

    # 2. Tính Beneish M-Score chuẩn
    m_score, is_beneish_fraud = calculate_beneish_mscore(
        dsri_val, gmi_val, aqi_val, sgi_val, depi_val, sgai_val, tata_val, lvgi_val
    )

    st.markdown("---")
    st.subheader("🎯 Kết Quả Đánh Giá Rủi Ro")

    res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 1.6])

    with res_col1:
        st.markdown("**Mô Hình Hồi Quy Logistic**")
        if is_logistic_fraud:
            st.markdown(
                f"""
                <div class="metric-card-danger">
                    <span class="badge-danger">NGUY CƠ GIAN LẬN CAO</span>
                    <h2 style='color:#dc2626; margin:8px 0;'>{prob_fraud*100:.1f}%</h2>
                    <p style='color:#6b7280; font-size:0.9rem; margin:0;'>Xác suất gian lận vượt ngưỡng quyết định ({threshold:.2f}). Cần kiểm toán chuyên sâu các khoản mục trọng yếu.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="metric-card-success">
                    <span class="badge-safe">AN TOÀN / RỦI RO THẤP</span>
                    <h2 style='color:#16a34a; margin:8px 0;'>{prob_fraud*100:.1f}%</h2>
                    <p style='color:#6b7280; font-size:0.9rem; margin:0;'>Xác suất gian lận dưới ngưỡng cảnh báo ({threshold:.2f}). Báo cáo tài chính chưa phát hiện dấu hiệu bất thường nghiêm trọng.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with res_col2:
        st.markdown("**Mô Hình Chuẩn Beneish (1999)**")
        if is_beneish_fraud:
            st.markdown(
                f"""
                <div class="metric-card-danger">
                    <span class="badge-danger">MANIPULATOR (M > -1.78)</span>
                    <h2 style='color:#dc2626; margin:8px 0;'>M = {m_score:.2f}</h2>
                    <p style='color:#6b7280; font-size:0.9rem; margin:0;'>Điểm M-Score vượt ngưỡng chuẩn -1.78. Doanh nghiệp có dấu hiệu thao túng lợi nhuận kế toán rõ rệt.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="metric-card-success">
                    <span class="badge-safe">NON-MANIPULATOR</span>
                    <h2 style='color:#16a34a; margin:8px 0;'>M = {m_score:.2f}</h2>
                    <p style='color:#6b7280; font-size:0.9rem; margin:0;'>Điểm M-Score &le; -1.78 nằm trong vùng an toàn theo mô hình Beneish gốc.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with res_col3:
        # Đồng hồ đo xác suất (Gauge Chart)
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_fraud * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Thước Đo Xác Suất Gian Lận (%)", "font": {"size": 15}},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444" if is_logistic_fraud else "#22c55e"},
                    "steps": [
                        {"range": [0, 30], "color": "#dcfce7"},
                        {"range": [30, 50], "color": "#fef9c3"},
                        {"range": [50, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": threshold * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Phân tích dấu hiệu cảnh báo đỏ (Red Flags)
    st.markdown("#### 🔍 Danh Sách Tín Hiệu Cảnh Báo Bất Thường (Red Flags)")
    red_flags = []
    for f in EXPECTED_FEATURES:
        val = input_data[f].iloc[0]
        ref = FEATURE_INFO[f]["red_flag"]
        if val > ref:
            red_flags.append(
                {
                    "Chỉ số": f,
                    "Tên chỉ số": FEATURE_INFO[f]["vn_name"],
                    "Giá trị hiện tại": f"{val:.3f}",
                    "Ngưỡng an toàn": f"< {ref:.2f}",
                    "Ý nghĩa cảnh báo": FEATURE_INFO[f]["desc"],
                }
            )

    if red_flags:
        rf_df = pd.DataFrame(red_flags)
        st.warning(f"⚠️ Phát hiện **{len(red_flags)}** chỉ số vượt ngưỡng an toàn thông thường:")
        st.dataframe(rf_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Toàn bộ 8 chỉ số đều nằm trong phạm vi an toàn thông thường.")

    # Biểu đồ Radar so sánh chỉ số với mức chuẩn
    st.markdown("#### 🕸️ Biểu Đồ Radar So Sánh Chỉ Số")
    radar_col1, radar_col2 = st.columns([2, 1])

    with radar_col1:
        # Chuẩn hóa để vẽ radar (giới hạn min 0, max 2.5 cho dễ nhìn)
        radar_categories = EXPECTED_FEATURES
        current_vals = [float(input_data[f].iloc[0]) for f in radar_categories]
        benchmark_safe = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.03, 1.0]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=current_vals + [current_vals[0]],
                theta=radar_categories + [radar_categories[0]],
                fill="toself",
                name="Doanh nghiệp đang xét",
                line=dict(color="#ef4444" if is_logistic_fraud else "#2563eb", width=2),
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=benchmark_safe + [benchmark_safe[0]],
                theta=radar_categories + [radar_categories[0]],
                fill="none",
                name="Mức chuẩn an toàn (1.0)",
                line=dict(color="#10b981", dash="dash", width=1.5),
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(2.0, max(current_vals))])),
            showlegend=True,
            height=380,
            margin=dict(l=30, r=30, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with radar_col2:
        st.markdown(
            """
            **Hướng dẫn quan sát biểu đồ:**
            - **Đường nét đứt xanh lá:** Mức tham chiếu an toàn thông thường.
            - **Đường màu (đang xét):** Nếu phình to vượt xa đường chuẩn ở các đỉnh **DSRI, SGI, AQI, TATA**, doanh nghiệp đang có dấu hiệu mở rộng bất thường hoặc dồn tích số ảo.
            - Đặc biệt chú ý **TATA** nếu mang giá trị dương lớn (> 0.08).
            """
        )

# ==============================================================================
# TAB 2: DỰ BÁO HÀNG LOẠT (FILE CSV / EXCEL)
# ==============================================================================
with tab2:
    st.subheader("Quét & Dự Báo Gian Lận Hàng Loạt Từ File")
    st.markdown(
        "Tải lên tệp CSV hoặc Excel chứa thông tin nhiều doanh nghiệp/nhiều niên độ để quét rủi ro tự động."
    )

    # Nút tải file mẫu template
    sample_df = pd.DataFrame(
        {
            "MA_CK": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "NAM": [2023, 2023, 2023, 2023, 2023],
            "DSRI": [1.02, 1.75, 0.95, 1.40, 0.88],
            "GMI": [0.98, 1.42, 1.05, 1.25, 0.92],
            "AQI": [0.85, 1.35, 0.90, 1.15, 0.80],
            "SGI": [1.12, 1.60, 1.08, 1.35, 1.02],
            "DEPI": [1.01, 1.28, 0.95, 1.10, 0.99],
            "SGAI": [0.94, 1.18, 1.02, 1.05, 0.90],
            "TATA": [0.03, 0.18, -0.02, 0.09, 0.01],
            "LVGI": [1.05, 1.45, 0.98, 1.20, 1.01],
        }
    )
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        st.download_button(
            label="📥 Tải Tệp CSV Mẫu",
            data=csv_buffer.getvalue(),
            file_name="mau_du_lieu_mscore.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_info:
        st.caption("Tệp dữ liệu cần chứa tối thiểu 8 cột: DSRI, GMI, AQI, SGI, DEPI, SGAI, TATA, LVGI (không phân biệt chữ hoa/thường).")

    uploaded_file = st.file_uploader(
        "Kéo thả hoặc chọn tệp tải lên (Hỗ trợ .CSV, .XLSX):",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)

            st.success(f"✅ Đã tải lên thành công: **{uploaded_file.name}** ({len(batch_df)} dòng)")

            # Chuẩn hóa tên cột
            col_map = {c: c.strip().upper() for c in batch_df.columns}
            batch_df = batch_df.rename(columns=col_map)

            missing_cols = [c for c in EXPECTED_FEATURES if c not in batch_df.columns]
            if missing_cols:
                st.error(f"❌ Tệp thiếu các cột bắt buộc: {missing_cols}. Vui lòng kiểm tra lại theo tệp mẫu.")
            else:
                # Ép kiểu dữ liệu số
                for c in EXPECTED_FEATURES:
                    batch_df[c] = pd.to_numeric(batch_df[c], errors="coerce")

                # Dự báo xác suất
                X_batch = batch_df[EXPECTED_FEATURES]
                batch_probs = pipeline.predict_proba(X_batch)[:, 1]
                batch_preds = (batch_probs >= threshold).astype(int)

                # Tính Beneish M-Score
                batch_mscores = []
                batch_beneish_flags = []
                for _, row in batch_df.iterrows():
                    m_val, m_flag = calculate_beneish_mscore(
                        row["DSRI"],
                        row["GMI"],
                        row["AQI"],
                        row["SGI"],
                        row["DEPI"],
                        row["SGAI"],
                        row["TATA"],
                        row["LVGI"],
                    )
                    batch_mscores.append(round(m_val, 3))
                    batch_beneish_flags.append("Gian lận" if m_flag else "An toàn")

                results_df = batch_df.copy()
                results_df["XAC_SUAT_GIAN_LAN"] = np.round(batch_probs, 4)
                results_df["DU_BAO_LOGISTIC"] = np.where(batch_preds == 1, "Rủi ro cao", "An toàn")
                results_df["BENEISH_MSCORE"] = batch_mscores
                results_df["DU_BAO_BENEISH"] = batch_beneish_flags

                # Đánh giá tổng hợp
                results_df["KET_LUAN_CHUNG"] = np.where(
                    (batch_preds == 1) | (np.array(batch_mscores) > -1.78),
                    "⚠️ CẢNH BÁO",
                    "✅ AN TOÀN",
                )

                # Thống kê tổng quan
                total_records = len(results_df)
                high_risk_count = (results_df["KET_LUAN_CHUNG"] == "⚠️ CẢNH BÁO").sum()
                safe_count = total_records - high_risk_count

                st.markdown("---")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Tổng Doanh Nghiệp Quét", f"{total_records:,}")
                m_col2.metric("Số Doanh Nghiệp Cảnh Báo", f"{high_risk_count:,}")
                m_col3.metric("Số Doanh Nghiệp An Toàn", f"{safe_count:,}")
                m_col4.metric("Tỷ Lệ Rủi Ro", f"{high_risk_count/total_records*100:.1f}%")

                # Bảng kết quả kèm bộ lọc
                st.markdown("#### Bảng Chi Tiết Kết Quả")
                filter_choice = st.radio(
                    "Lọc kết quả hiển thị:",
                    ["Tất cả", "Chỉ hiển thị Cảnh báo rủi ro (⚠️)", "Chỉ hiển thị An toàn (✅)"],
                    horizontal=True,
                )

                display_df = results_df.copy()
                if filter_choice == "Chỉ hiển thị Cảnh báo rủi ro (⚠️)":
                    display_df = display_df[display_df["KET_LUAN_CHUNG"] == "⚠️ CẢNH BÁO"]
                elif filter_choice == "Chỉ hiển thị An toàn (✅)":
                    display_df = display_df[display_df["KET_LUAN_CHUNG"] == "✅ AN TOÀN"]

                st.dataframe(
                    display_df.style.map(
                        lambda v: "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
                        if v == "⚠️ CẢNH BÁO"
                        else ("background-color: #dcfce7; color: #166534;" if v == "✅ AN TOÀN" else ""),
                        subset=["KET_LUAN_CHUNG"],
                    ),
                    use_container_width=True,
                )

                # Nút tải kết quả ra Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    results_df.to_excel(writer, index=False, sheet_name="Ket_qua_du_bao")
                excel_data = output.getvalue()

                st.download_button(
                    label="📊 Tải Kết Quả Dự Báo Về Máy (.XLSX)",
                    data=excel_data,
                    file_name="Ket_qua_du_bao_gian_lan_BCTC.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi xử lý tệp: {e}")
    else:
        st.info("💡 Chưa có tệp nào được chọn. Bạn có thể bấm 'Tải Tệp CSV Mẫu' ở trên để kiểm tra thử tính năng.")

# ==============================================================================
# TAB 3: HIỆU NĂNG & HỆ SỐ MÔ HÌNH
# ==============================================================================
with tab3:
    st.subheader("Đánh Giá Hiệu Năng Mô Hình Học Máy (Test Set 20%)")

    # Tính toán các chỉ số trên test set
    y_test_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    # Hiển thị thẻ metrics
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    kpi_col1.metric("Accuracy (Độ chính xác)", f"{acc*100:.1f}%")
    kpi_col2.metric("Precision (Độ chuẩn xác)", f"{prec*100:.1f}%")
    kpi_col3.metric("Recall (Độ nhạy)", f"{rec*100:.1f}%")
    kpi_col4.metric("Specificity (Độ đặc hiệu)", f"{spec*100:.1f}%")
    kpi_col5.metric("F1-Score", f"{f1:.3f}")
    kpi_col6.metric("ROC-AUC", f"{auc:.3f}")

    st.caption(
        f"*Lưu ý: Kết quả trên thay đổi trực tiếp theo Ngưỡng quyết định (Threshold = {threshold:.2f}) thiết lập ở thanh bên.*"
    )

    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Ma trận nhầm lẫn
        cm_data = [[tn, fp], [fn, tp]]
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Dự Báo Của Mô Hình", y="Thực Tế", color="Số Lượng"),
            x=["Không gian lận (0)", "Gian lận (1)"],
            y=["Không gian lận (0)", "Gian lận (1)"],
            text_auto=True,
            color_continuous_scale="Blues",
            title="Ma Trận Nhầm Lẫn (Confusion Matrix)",
        )
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    with chart_col2:
        # Đường cong ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Mô hình (AUC = {auc:.3f})", line=dict(color="#2563eb", width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ngẫu nhiên", line=dict(color="gray", dash="dash")))
        fig_roc.update_layout(
            title="Đường Cong ROC (Receiver Operating Characteristic)",
            xaxis_title="Tỷ lệ Cảnh báo nhầm (False Positive Rate)",
            yaxis_title="Tỷ lệ Bắt trúng (True Positive Rate / Recall)",
            height=350,
            showlegend=True,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    st.subheader("Trọng Số & Ý Nghĩa Từng Biến Trong Mô Hình")

    # Bảng hệ số
    st.markdown("#### Bảng Hệ Số Hồi Quy Logistic (Trên thang đo gốc)")
    st.dataframe(
        coef_df.style.format(
            {
                "Hệ số gốc (Beta)": "{:.4f}",
                "Hệ số chuẩn hóa": "{:.4f}",
                "Odds Ratio (1 đv)": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Phương trình toán học
    terms = [f"({b:.4f} × {v})" for v, b in zip(coef_df["Biến số"], coef_df["Hệ số gốc (Beta)"])]
    eq_str = f"Z = {intercept_raw:.4f} + " + " + ".join(terms)

    st.markdown("**Phương trình hồi quy tổng quát:**")
    st.code(eq_str, language="text")
    st.markdown(
        """
        **Xác suất gian lận được tính theo hàm Sigmoid:**
        $$P(\\text{FRAUD}=1) = \\frac{1}{1 + e^{-Z}}$$
        """
    )

    # Biểu đồ thanh biểu diễn trọng số
    fig_bar = px.bar(
        coef_df.sort_values(by="Hệ số chuẩn hóa", ascending=True),
        x="Hệ số chuẩn hóa",
        y="Biến số",
        orientation="h",
        color="Hệ số chuẩn hóa",
        color_continuous_scale="RdBu_r",
        title="Tầm Quan Trọng Chuẩn Hóa Của Các Biến (Standardized Coefficients)",
        text="Hệ số chuẩn hóa",
    )
    fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

# ==============================================================================
# TAB 4: CẨM NANG 8 CHỈ SỐ BENEISH
# ==============================================================================
with tab4:
    st.subheader("📚 Cẩm Nang Tra Cứu 8 Chỉ Số Beneish M-Score")
    st.markdown(
        """
        Mô hình **Beneish M-Score** được công bố bởi Giáo sư **Messod Beneish** (1999). 
        Đây là một trong những công cụ toán học - kế toán pháp y kinh điển nhất được sử dụng bởi các kiểm toán viên, 
        Ủy ban Chứng khoán Hoa Kỳ (SEC), và các nhà quản lý quỹ phòng hộ (tiêu biểu là việc phát hiện ra vụ bê bối Enron trước khi tập đoàn này sụp đổ).
        """
    )

    guide_items = [
        {
            "Chỉ số": "1. DSRI",
            "Tên đầy đủ": "Days Sales in Receivables Index",
            "Tiếng Việt": "Số ngày phải thu khách hàng",
            "Công thức": "(Khoản phải thu_t / Doanh thu_t) ÷ (Khoản phải thu_t-1 / Doanh thu_t-1)",
            "Ngưỡng rủi ro": "> 1.30",
            "Dấu hiệu cảnh báo": "Phải thu tăng vượt trội so với doanh thu. Có thể doanh nghiệp đang đẩy hàng khống cho đại lý, ghi nhận doanh thu ảo trước kỳ quyết toán hoặc giảm tiêu chuẩn cấp tín dụng để đạt kế hoạch.",
        },
        {
            "Chỉ số": "2. GMI",
            "Tên đầy đủ": "Gross Margin Index",
            "Tiếng Việt": "Tỷ suất lãi gộp suy giảm",
            "Công thức": "[(Doanh thu_t-1 - Giá vốn_t-1) / Doanh thu_t-1] ÷ [(Doanh thu_t - Giá vốn_t) / Doanh thu_t]",
            "Ngưỡng rủi ro": "> 1.20",
            "Dấu hiệu cảnh báo": "Lưu ý: Nếu biên lợi nhuận gộp năm nay xấu đi so với năm trước thì GMI sẽ > 1. Khi hoạt động cốt lõi sa sút, ban giám đốc chịu áp lực lớn trong việc gian lận sổ sách để che giấu kết quả tồi tệ.",
        },
        {
            "Chỉ số": "3. AQI",
            "Tên đầy đủ": "Asset Quality Index",
            "Tiếng Việt": "Chất lượng tài sản",
            "Công thức": "[1 - (Tài sản ngắn hạn_t + TSCĐ_t + Chứng khoán_t) / Tổng TS_t] ÷ [Tương tự kỳ t-1]",
            "Ngưỡng rủi ro": "> 1.25",
            "Dấu hiệu cảnh báo": "Tài sản phi vật chất/vô hình dở dang tăng vọt. Dấu hiệu điển hình của việc vốn hóa các chi phí phát sinh trong kỳ vào tài sản thay vì đưa vào chi phí hợp lý để giảm bớt khoản lỗ.",
        },
        {
            "Chỉ số": "4. SGI",
            "Tên đầy đủ": "Sales Growth Index",
            "Tiếng Việt": "Tăng trưởng doanh thu",
            "Công thức": "Doanh thu kỳ t ÷ Doanh thu kỳ t-1",
            "Ngưỡng rủi ro": "> 1.30",
            "Dấu hiệu cảnh báo": "Tăng trưởng tự thân không phải là xấu, nhưng các công ty có tốc độ tăng trưởng quá nóng thường dễ gục ngã và có động cơ gian lận khi thị trường bão hòa nhằm giữ hình ảnh cổ phiếu.",
        },
        {
            "Chỉ số": "5. DEPI",
            "Tên đầy đủ": "Depreciation Index",
            "Tiếng Việt": "Tỷ lệ khấu hao tài sản",
            "Công thức": "Tỷ lệ khấu hao kỳ t-1 ÷ Tỷ lệ khấu hao kỳ t",
            "Ngưỡng rủi ro": "> 1.15",
            "Dấu hiệu cảnh báo": "Tỷ lệ khấu hao năm nay thấp hơn năm trước. Cho thấy công ty đã thay đổi phương pháp khấu hao, tăng thời gian ước tính sử dụng hữu ích của tài sản để giảm chi phí khấu hao trong kỳ.",
        },
        {
            "Chỉ số": "6. SGAI",
            "Tên đầy ""đủ": "Sales, General & Admin Expenses Index",
            "Tiếng Việt": "Chi phí bán hàng & Quản lý",
            "Công thức": "(Chi phí SG&A_t / Doanh thu_t) ÷ (Chi phí SG&A_t-1 / Doanh thu_t-1)",
            "Ngưỡng rủi ro": "> 1.20",
            "Dấu hiệu cảnh báo": "Chi phí điều hành tăng nhanh hơn tăng trưởng doanh thu. Cho thấy hiệu quả hoạt động kém đi, biên lợi nhuận bị bào mòn.",
        },
        {
            "Chỉ số": "7. TATA",
            "Tên đầy đủ": "Total Accruals to Total Assets",
            "Tiếng Việt": "Biến dồn tích trên tổng tài sản",
            "Công thức": "(Lợi nhuận thuần từ HĐKD_t - Dòng tiền thuần từ HĐKD CFO_t) ÷ Tổng tài sản_t",
            "Ngưỡng rủi ro": "> 0.08",
            "Dấu hiệu cảnh báo": "Đây là chỉ số quan trọng bậc nhất! Lợi nhuận sổ sách kế toán dương nhưng dòng tiền mặt âm (Accruals cao). Doanh nghiệp có thể đang 'lãi giả, lỗ thật'.",
        },
        {
            "Chỉ số": "8. LVGI",
            "Tên đầy đủ": "Leverage Index",
            "Tiếng Việt": "Chỉ số đòn bẩy tài chính",
            "Công thức": "(Tổng nợ vay_t / Tổng tài sản_t) ÷ (Tổng nợ vay_t-1 / Tổng tài sản_t-1)",
            "Ngưỡng rủi ro": "> 1.25",
            "Dấu hiệu cảnh báo": "Nợ vay tăng nhanh so với tài sản, tăng rủi ro thanh khoản và vi phạm các điều khoản cam kết với ngân hàng, thúc đẩy hành vi gian dối sổ sách.",
        },
    ]

    guide_df = pd.DataFrame(guide_items)
    st.dataframe(guide_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        """
        ### 💡 Lời Khuyên Cho Kiểm Toán Viên & Nhà Đầu Tư
        1. **Không kết luận vội vã chỉ dựa vào 1 chỉ số**: Một chỉ số vượt ngưỡng có thể bắt nguồn từ chiến lược kinh doanh đặc thù (ví dụ công ty phần mềm có ít TSCĐ, công ty bán lẻ có chu kỳ thu tiền khác).
        2. **Đặc biệt cảnh giác khi cả TATA và DSRI cùng tăng mạnh**: Đây là dấu hiệu kinh điển của việc ghi nhận doanh thu ảo mà tiền mặt không bao giờ về tài khoản.
        3. **Đối chiếu chéo 2 mô hình**: Kết hợp cả xác suất của Hồi quy Logistic (huấn luyện trên bối cảnh dữ liệu thực tế) và điểm Beneish M-Score chuẩn để đưa ra nhận định khách quan nhất.
        """
    )
