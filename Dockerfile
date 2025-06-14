# Sử dụng base image Python 3.11.9 (hoặc 3.11-slim-buster)
FROM python:3.11-slim-buster

# Cài đặt các công cụ hệ thống cần thiết (build-essential cho biên dịch, libgomp1 cho numpy/scipy)
# và các tài nguyên NLTK
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    apt-utils \
    libgomp1 \
    # NLTK data (quan trọng cho preprocess_text)
    # Các lệnh tải dữ liệu NLTK chạy trong build phase của Docker
    && python -m nltk.downloader stopwords punkt \
    && rm -rf /var/lib/apt/lists/* # Xóa cache apt để giảm kích thước image

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép requirements.txt vào container và cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install gunicorn # Đảm bảo gunicorn được cài đặt

# Sao chép tất cả các file còn lại của dự án vào container
COPY . .

# Mở cổng mà ứng dụng Flask sẽ lắng nghe bên trong container
EXPOSE 5000

# Lệnh để chạy ứng dụng bằng Gunicorn khi container khởi động
# 0.0.0.0:5000 là địa chỉ mà Gunicorn sẽ lắng nghe
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.app:app"] # Thay api.app:app nếu app.py ở thư mục gốc