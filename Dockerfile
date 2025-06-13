# Sử dụng một base image Python ổn định và tương thích với TensorFlow
# Python 3.9 hoặc 3.10 thường là lựa chọn tốt nhất cho TF
# "slim-buster" là bản rút gọn, giúp giảm kích thước image
FROM python:3.11-slim-buster

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép requirements.txt vào container
COPY requirements.txt .

# Cài đặt các thư viện Python
# --no-cache-dir để giảm kích thước image
# && python -m spacy download en_core_web_sm để tải mô hình spaCy trong quá trình build
# && pip install gunicorn nếu bạn chưa có trong requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && pip install gunicorn # Đảm bảo gunicorn được cài đặt

# Sao chép tất cả các file còn lại của dự án vào container
# Đảm bảo các file .h5 và .pkl của bạn trong thư mục models/ cũng được sao chép
COPY . .

# Mở cổng mà ứng dụng của bạn sẽ lắng nghe (mặc định Flask là 5000)
EXPOSE 5000

# Lệnh để chạy ứng dụng khi container khởi động
# Gunicorn lắng nghe trên tất cả các địa chỉ IP trên cổng 5000 (mà Spaces sẽ ánh xạ)
# app:app nghĩa là tìm biến 'app' trong file 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]