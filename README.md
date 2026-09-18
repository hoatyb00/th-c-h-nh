# ⚖️ HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH (BCTC) VỚI STREAMLIT

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Ứng dụng Web phát hiện và cảnh báo sớm nguy cơ gian lận, thao túng báo cáo tài chính (BCTC) dựa trên sự kết hợp giữa **Mô hình Hồi quy Logistic (Machine Learning)** và **Mô hình Beneish M-Score chuẩn quốc tế (1999)**.

---

## 📌 1. TỔNG QUAN DỰ ÁN

Gian lận báo cáo tài chính là một trong những rủi ro lớn nhất đối với kiểm toán viên, chuyên viên thẩm định tín dụng, cơ quan quản lý và các nhà đầu tư. Dự án này số hóa phương pháp **Kế toán Pháp y (Forensic Accounting)** thành một công cụ trực tuyến tương tác cao:
- **Dữ liệu huấn luyện**: 500 quan sát doanh nghiệp (`MScore_data.csv`) với 8 chỉ số thành phần và biến mục tiêu `FRAUD_FLAG` (0: Không gian lận, 1: Gian lận).
- **Mô hình cốt lõi**:
  1. **Logistic Regression Pipeline**: Xử lý khuyết thiếu (Median Imputer) $\rightarrow$ Chuẩn hóa dữ liệu (StandardScaler) $\rightarrow$ Hồi quy Logistic (`liblinear`).
  2. **Beneish M-Score 8 biến (1999)**: Tính điểm $M$ theo trọng số chuẩn mực của GS. Messod Beneish, phân loại rủi ro theo ngưỡng $M > -1.78$.

---

## ✨ 2. TÍNH NĂNG NỔI BẬT

1. **🔮 Dự báo doanh nghiệp đơn lẻ (Single Prediction)**:
   - Nhập liệu trực quan cho 8 chỉ số với giải thích chi tiết và gợi ý khoảng giá trị chuẩn.
   - Nút chọn kịch bản mẫu nhanh: *Doanh nghiệp an toàn* vs. *Doanh nghiệp rủi ro cao*.
   - Đánh giá song song bằng **Xác suất Logistic** và **Điểm Beneish M-Score**.
   - Thước đo mức độ rủi ro dạng đồng hồ (Gauge Chart) và biểu đồ mạng nhện (Radar Chart).
   - Tự động liệt kê danh sách các **Tín hiệu cảnh báo đỏ (Red Flags)** vượt ngưỡng an toàn.

2. **📁 Quét & Dự báo hàng loạt (Batch Processing)**:
   - Tải lên tệp dữ liệu `.csv` hoặc `.xlsx` chứa nhiều doanh nghiệp/nhiều niên độ.
   - Cung cấp nút **Tải tệp mẫu (Sample CSV)** chuẩn cấu trúc.
   - Tự động tính toán xác suất, phân loại nhãn và tổng hợp tỷ lệ rủi ro toàn danh mục.
   - Bộ lọc xem riêng các doanh nghiệp thuộc nhóm rủi ro cao.
   - Xuất kết quả phân tích đầy đủ ra tệp Excel (`.xlsx`).

3. **📊 Đánh giá mô hình & Tầm quan trọng của chỉ số**:
   - Thống kê đầy đủ các chỉ số hiệu năng trên tập kiểm định Test set (20%): Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC.
   - **Thanh trượt ngưỡng quyết định (Threshold Slider)**: Linh hoạt thay đổi ngưỡng để quan sát sự đánh đổi giữa bắt nhầm và bỏ lọt gian lận.
   - Biểu đồ Ma trận nhầm lẫn (Confusion Matrix) và Đường cong ROC trực quan.
   - Bảng hệ số hồi quy gốc, Odds Ratio và biểu đồ tầm quan trọng của các chỉ số.

4. **📚 Cẩm nang 8 chỉ số Beneish**:
   - Tra cứu công thức kế toán, ý nghĩa kinh tế và tín hiệu cảnh báo chuyên sâu của cả 8 chỉ số (DSRI, GMI, AQI, SGI, DEPI, SGAI, TATA, LVGI).

---

## 🗂️ 3. CẤU TRÚC THƯ MỤC

```text
tao app thuc hanh/
│
├── app.py                              # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt                    # Danh sách thư viện Python cần thiết
├── README.md                           # Hướng dẫn chi tiết dự án & triển khai
├── MScore_data.csv                     # Tập dữ liệu huấn luyện (500 quan sát)
└── logistic_mscore_google_colab.py     # Script gốc từ Google Colab
```

---

## 🚀 4. HƯỚNG DẪN CÀI ĐẶT & CHẠY TẠI MÁY CÁ NHÂN (LOCAL)

### Yêu cầu tiên quyết:
- Máy tính đã cài đặt Python (khuyến nghị từ bản 3.10 đến 3.12).
- Trình thông dịch lệnh PowerShell hoặc Terminal/Command Prompt.

### Các bước thực hiện:

1. **Mở thư mục dự án**:
   ```bash
   cd "c:\Users\Admin\Desktop\tao app thuc hanh"
   ```

2. **Tạo và kích hoạt môi trường ảo (Virtual Environment)**:
   - Trên Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   *(Nếu gặp lỗi script execution policy trên PowerShell, chạy lệnh: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` rồi kích hoạt lại)*

3. **Cài đặt các thư viện cần thiết**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Khởi chạy ứng dụng Streamlit**:
   ```bash
   streamlit run app.py
   ```
   Trình duyệt web sẽ tự động mở trang web tại địa chỉ: `http://localhost:8501`.

---

## 🌐 5. HƯỚNG DẪN ĐẨY LÊN GITHUB & DEPLOY TRÊN STREAMLIT CLOUD

### Bước 1: Đẩy mã nguồn lên GitHub

1. Truy cập [GitHub](https://github.com) và tạo một Repository mới (ví dụ đặt tên: `fraud-detection-bctc`), chọn chế độ **Public**.
2. Tại thư mục máy tính của bạn, mở PowerShell và chạy chuỗi lệnh sau:
   ```bash
   # Khởi tạo git repository
   git init

   # Thêm tất cả các file cần thiết
   git add app.py requirements.txt README.md MScore_data.csv

   # Commit phiên bản đầu tiên
   git commit -m "Khoi tao Streamlit Web App du bao gian lan BCTC"

   # Đổi tên nhánh chính thành main
   git branch -M main

   # Liên kết với repository trên GitHub (thay username và repo của bạn)
   git remote add origin https://github.com/<USERNAME-CUA-BAN>/fraud-detection-bctc.git

   # Đẩy code lên GitHub
   git push -u origin main
   ```

### Bước 2: Triển khai (Deploy) miễn phí trên Streamlit Community Cloud

1. Truy cập vào trang quản lý [Streamlit Community Cloud](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Bấm vào nút **"Create app"** (hoặc **"New app"**).
3. Điền các thông tin triển khai:
   - **Repository**: Chọn repo bạn vừa tạo (`<USERNAME>/fraud-detection-bctc`).
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Bấm **"Deploy!"**.
5. Trong vòng 1–2 phút, Streamlit Cloud sẽ tự động cài đặt các thư viện từ `requirements.txt` và cung cấp cho bạn một đường link công khai dạng:
   `https://fraud-detection-bctc-<ma-ung-dung>.streamlit.app`
   Bạn có thể gửi link này cho bạn bè, giảng viên, đồng nghiệp hoặc đối tác sử dụng trực tiếp trên mọi thiết bị (máy tính, điện thoại, tablet) mà không cần cài đặt gì thêm!

---

## 📊 6. TÓM TẮT 8 CHỈ SỐ BENEISH M-SCORE

| Chỉ số | Tên tiếng Anh | Tên tiếng Việt | Ngưỡng rủi ro | Dấu hiệu cảnh báo |
| :--- | :--- | :--- | :---: | :--- |
| **DSRI** | Days Sales in Receivables Index | Số ngày phải thu khách hàng | `> 1.30` | Phải thu tăng nhanh bất thường so với doanh thu; rủi ro ghi nhận doanh thu ảo. |
| **GMI** | Gross Margin Index | Tỷ suất lãi gộp suy giảm | `> 1.20` | Biên lãi gộp sa sút; động cơ làm đẹp số liệu để duy trì kỳ vọng thị trường. |
| **AQI** | Asset Quality Index | Chất lượng tài sản | `> 1.25` | Tỷ trọng tài sản vô hình/chi phí dở dang tăng vọt; vốn hóa chi phí để giảm lỗ. |
| **SGI** | Sales Growth Index | Tăng trưởng doanh thu | `> 1.30` | Doanh thu tăng trưởng quá nóng; dễ xảy ra gian lận khi tốc độ tăng chững lại. |
| **DEPI** | Depreciation Index | Tỷ lệ khấu hao | `> 1.15` | Tỷ lệ khấu hao giảm; kéo dài thời gian khấu hao tài sản để thổi phồng lợi nhuận. |
| **SGAI** | SGA Expense Index | Chi phí bán hàng & QLDN | `> 1.20` | Chi phí vận hành tăng nhanh hơn doanh thu; hiệu quả hoạt động kém đi. |
| **TATA** | Total Accruals to Total Assets | Biến dồn tích / Tổng TS | `> 0.08` | Lợi nhuận kế toán không có dòng tiền mặt bảo đảm ("Lãi giả, lỗ thật"). |
| **LVGI** | Leverage Index | Đòn bẩy tài chính | `> 1.25` | Tỷ lệ nợ vay tăng nhanh; áp lực thanh khoản và vi phạm điều khoản tín dụng. |

---

## ⚖️ 7. KHUYẾN CÁO MIỄN TRỪ TRÁCH NHIỆM (DISCLAIMER)

Mô hình này là công cụ hỗ trợ rà soát rủi ro ban đầu (Screening Tool). Kết quả từ mô hình:
- **Không cấu thành kết luận pháp lý** về việc doanh nghiệp có hành vi vi phạm pháp luật.
- Các chuyên viên phân tích và kiểm toán viên cần kết hợp phỏng vấn, kiểm tra chứng từ gốc và phân tích định tính trước khi đưa ra kết luận chính thức.

---

## 👨‍💻 TÁC GIẢ & HỖ TRỢ
- Dự án được xây dựng phục vụ nghiên cứu, học tập và ứng dụng thực tiễn trong phân tích tài chính doanh nghiệp.
- Mọi thắc mắc hoặc đề xuất tính năng, xin vui lòng tạo **Issue** trên repository GitHub.
