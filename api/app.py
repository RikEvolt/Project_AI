# from flask import Flask, request, jsonify
# from flasgger import Swagger, swag_from
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np
# import pickle
# import requests
# from bs4 import BeautifulSoup
# from dotenv import load_dotenv
# import os
# from flask_limiter import Limiter
# from flask_limiter.util import get_remote_address

# # Import các hàm từ utils.py
# from utils import (
#     stable_randint,
#     extract_keywords_tfidf,
#     analyze_sections,
#     get_simulated_pagespeed_score,
#     get_simulated_top_keywords,
#     get_simulated_keyword_rank,
#     get_simulated_search_trends,
#     preprocess_text # Import thêm hàm này để tiền xử lý văn bản cho predict
# )

# # Tải biến môi trường
# load_dotenv()
# # Ví dụ: Nếu bạn có API key nào đó sau này muốn dùng (hiện tại không dùng)
# # SOME_API_KEY = os.getenv("SOME_API_KEY")

# # Khởi tạo app Flask & Swagger
# app = Flask(__name__)
# Swagger(app)

# # Khởi tạo Limiter (giới hạn request để tránh bị DDoS hoặc quá tải)
# limiter = Limiter(
#     get_remote_address,
#     app=app,
#     default_limits=["30 per minute", "1000 per day"]  # Giới hạn 30 request/phút, 1000 request/ngày mỗi IP
# )

# # Load mô hình và tokenizer (giả định đã được huấn luyện và lưu)
# # Nếu bạn chưa có, bạn cần huấn luyện một mô hình NLP và tokenizer trước
# # Để chạy được phần này, bạn cần có 2 file này trong thư mục 'models'
# try:
#     model = tf.keras.models.load_model("models/keyword_model.h5")
#     with open("models/Tokenizer.pkl", "rb") as f:
#         tokenizer = pickle.load(f)
#     print("Mô hình và tokenizer đã được tải thành công.")
# except (OSError, FileNotFoundError) as e:
#     print(f"Lỗi khi tải mô hình hoặc tokenizer: {e}")
#     print("Vui lòng đảm bảo các file 'keyword_model.h5' và 'Tokenizer.pkl' có trong thư mục 'models/'.")
#     print("Nếu chưa có, bạn cần huấn luyện một mô hình và lưu lại trước.")
#     # Tạo các đối tượng giả để app vẫn chạy được nhưng các chức năng predict sẽ không hoạt động đúng
#     model = None
#     tokenizer = None

# # Định nghĩa độ dài tối đa cho padding (phải khớp với khi huấn luyện mô hình)
# MAX_SEQUENCE_LENGTH = 200 # Giả sử độ dài này


# # ========= ROUTES API ==========

# @app.route("/predict", methods=["POST"])
# @limiter.limit("10 per minute") # Giới hạn riêng cho route này
# @swag_from({
#     'parameters': [
#         {
#             'name': 'text',
#             'in': 'body',
#             'required': True,
#             'schema': {
#                 'type': 'object',
#                 'properties': {
#                     'text': {'type': 'string', 'example': 'Improve your website SEO using keyword optimization'}
#                 }
#             }
#         }
#     ],
#     'responses': {
#         200: {
#             'description': 'Kết quả dự đoán (có phải là văn bản thân thiện SEO hay không)'
#         },
#         400: {
#             'description': 'Lỗi đầu vào hoặc mô hình không khả dụng'
#         }
#     }
# })
# def predict():
#     if model is None or tokenizer is None:
#         return jsonify({"error": "Mô hình hoặc tokenizer chưa được tải. Vui lòng kiểm tra lại file mô hình."}), 500

#     data = request.get_json()
#     text = data.get("text", "")
#     if not text:
#         return jsonify({"error": "Không có văn bản để dự đoán"}), 400

#     # Tiền xử lý văn bản tương tự như khi huấn luyện
#     processed_text = preprocess_text(text)
#     if not processed_text:
#         return jsonify({"prediction": 0, "confidence": 0.0, "message": "Văn bản quá ngắn hoặc không có nội dung sau khi tiền xử lý."}), 200

#     sequence = tokenizer.texts_to_sequences([processed_text])
#     padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    
#     # Dự đoán
#     pred = model.predict(padded)[0][0]
    
#     # Kết quả
#     # Giả sử mô hình của bạn trả về một giá trị từ 0 đến 1,
#     # nơi giá trị càng cao thì càng thân thiện với SEO
#     is_seo_friendly = int(pred > 0.5) # Ngưỡng 0.5 để phân loại
#     confidence = float(pred)

#     return jsonify({
#         "prediction": is_seo_friendly,
#         "confidence": confidence,
#         "message": "Thân thiện với SEO" if is_seo_friendly else "Chưa thân thiện với SEO",
#         "raw_prediction_score": confidence
#     })


# @app.route("/analyze", methods=["POST"])
# @limiter.limit("5 per minute")
# @swag_from({
#     'parameters': [
#         {
#             'name': 'url',
#             'in': 'body',
#             'required': True,
#             'schema': {
#                 'type': 'object',
#                 'properties': {
#                     'url': {'type': 'string', 'example': 'https://example.com'}
#                 }
#             }
#         }
#     ],
#     'responses': {
#         200: {'description': 'Kết quả phân tích SEO của một URL'},
#         400: {'description': 'Lỗi khi phân tích URL'}
#     }
# })
# def analyze():
#     data = request.get_json()
#     url = data.get("url", "")
#     if not url:
#         return jsonify({"error": "URL không được cung cấp"}), 400

#     try:
#         # Lấy nội dung trang
#         response = requests.get(url, timeout=10) # Tăng timeout
#         response.raise_for_status() # Kiểm tra lỗi HTTP (ví dụ: 404, 500)
#         soup = BeautifulSoup(response.text, "html.parser")
        
#         # Trích xuất toàn bộ văn bản thuần túy từ trang
#         full_text = soup.get_text(separator=' ', strip=True)
#         if len(full_text) < 100:
#             raise Exception("Không đủ nội dung để phân tích (ít hơn 100 ký tự).")

#         # Tiền xử lý văn bản cho mô hình dự đoán (nếu mô hình NLP được dùng)
#         processed_text_for_model = preprocess_text(full_text)
        
#         # Dự đoán điểm SEO của nội dung (dùng mô hình nếu có)
#         web_score = 0
#         content_score = 0
#         if model and tokenizer and processed_text_for_model:
#             sequence = tokenizer.texts_to_sequences([processed_text_for_model])
#             padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
#             pred = model.predict(padded)[0][0]
#             web_score = int(pred * 100) # Điểm từ 0-100 dựa trên dự đoán mô hình
#             content_score = int(min(95, web_score + stable_randint(-5, 5, url + "content")))
#         else:
#             # Nếu không có mô hình, tạo điểm giả lập hoặc điểm mặc định
#             web_score = stable_randint(40, 80, url + "web_score") # Điểm ngẫu nhiên ổn định
#             content_score = stable_randint(50, 90, url + "content_score")


#         # Các thông số SEO khác
#         keywords_top10 = int((web_score / 100 * 20) + stable_randint(5, 10, url + "kw_top10")) # Ước tính số từ khóa top 10
#         backlinks = int(500 + web_score * 50) # Ước tính số lượng backlink (giả lập)


#         keywords_extracted = extract_keywords_tfidf(full_text) # Trích xuất từ khóa bằng TF-IDF
#         section_analysis = analyze_sections(soup) # Phân tích tiêu đề, H1, meta description
#         pagespeed = get_simulated_pagespeed_score(url) # Điểm Pagespeed giả lập

#         # Tổng hợp điểm
#         overall_score = (
#             (web_score * 0.4) + # 40% từ điểm nội dung/model
#             (pagespeed * 0.2) + # 20% từ tốc độ
#             (section_analysis["title_score"] * 0.15) + # 15% từ tiêu đề
#             (section_analysis["h1_score"] * 0.15) + # 15% từ H1
#             (section_analysis["meta_description_score"] * 0.05) + # 5% từ meta desc
#             (section_analysis["https_score"] * 0.05) # 5% từ HTTPS
#         ) / 100 # Chia 100 để chuẩn hóa lại nếu tổng trọng số không phải 100
        
#         overall_score = int(min(100, max(0, overall_score))) # Đảm bảo điểm nằm trong khoảng 0-100

#         return jsonify({
#             "url": url,
#             "overall_seo_score": overall_score,
#             "content_quality_score": web_score,
#             "content_readability_score": content_score, # Giả lập điểm đọc hiểu
#             "estimated_keywords_top10": keywords_top10,
#             "estimated_backlinks": backlinks,
#             "extracted_keywords": keywords_extracted,
#             "onpage_analysis": {
#                 "title": section_analysis["title"],
#                 "title_score": section_analysis["title_score"],
#                 "h1_tags": section_analysis["h1_tags"],
#                 "h1_score": section_analysis["h1_score"],
#                 "meta_description": section_analysis["meta_description"],
#                 "meta_description_score": section_analysis["meta_description_score"],
#                 "is_https": section_analysis["is_https"],
#                 "https_score": section_analysis["https_score"],
#                 # Bạn có thể thêm: alt_text_ratio, internal_link_count, broken_links_count (cần cào sâu hơn)
#             },
#             "pagespeed_score": pagespeed,
#             "summary": f"Trang web {url} có điểm SEO tổng thể {overall_score}/100. Điểm chất lượng nội dung {web_score}/100, tốc độ {pagespeed}/100."
#         })

#     except requests.exceptions.RequestException as req_err:
#         return jsonify({
#             "error": f"Lỗi khi truy cập URL: {req_err}",
#             "summary": f"Không thể kết nối hoặc tải nội dung từ URL: {url}"
#         }), 400
#     except Exception as e:
#         return jsonify({
#             "error": str(e),
#             "summary": f"Lỗi khi phân tích URL: {url}"
#         }), 400


# @app.route("/top_keywords", methods=["GET"])
# @limiter.limit("20 per minute")
# @swag_from({
#     'parameters': [
#         {'name': 'geo', 'in': 'query', 'type': 'string', 'required': False, 'default': 'VN', 'enum': ['VN', 'US', 'JP', 'KR', 'IN']},
#         {'name': 'count', 'in': 'query', 'type': 'integer', 'required': False, 'default': 20}
#     ],
#     'responses': {
#         200: {
#             'description': 'Danh sách từ khóa thịnh hành giả lập theo khu vực'
#         }
#     }
# })
# def top_keywords():
#     geo = request.args.get("geo", "VN").upper()
#     count = request.args.get("count", type=int, default=20)
    
#     keywords = get_simulated_top_keywords(geo, count)
#     return jsonify({"geo": geo, "count": len(keywords), "keywords": keywords})


# @app.route("/keyword_rank", methods=["GET"])
# @limiter.limit("10 per minute")
# @swag_from({
#     'parameters': [
#         {'name': 'keyword', 'in': 'query', 'type': 'string', 'required': True, 'example': 'SEO tools'},
#         {'name': 'domain', 'in': 'query', 'type': 'string', 'required': True, 'example': 'example.com'},
#         {'name': 'max_pages', 'in': 'query', 'type': 'integer', 'required': False, 'default': 3}
#     ],
#     'responses': {
#         200: {
#             'description': 'Thứ hạng từ khóa giả lập của domain cho một từ khóa'
#         }
#     }
# })
# def keyword_rank():
#     keyword = request.args.get("keyword")
#     domain = request.args.get("domain")
#     max_pages = request.args.get("max_pages", type=int, default=3)

#     if not keyword or not domain:
#         return jsonify({"error": "Thiếu keyword hoặc domain"}), 400

#     rank = get_simulated_keyword_rank(keyword, domain, max_pages)
#     return jsonify({
#         "keyword": keyword,
#         "domain": domain,
#         "rank": rank,
#         "message": "Không tìm thấy trong top kết quả được mô phỏng" if rank == -1 else "Tìm thấy trong kết quả được mô phỏng"
#     })


# @app.route("/search_trends", methods=["GET"])
# @limiter.limit("10 per minute")
# @swag_from({
#     'parameters': [
#         {'name': 'q', 'in': 'query', 'type': 'string', 'required': True, 'example': 'machine learning'},
#         {'name': 'geo', 'in': 'query', 'type': 'string', 'required': False, 'default': 'VN'}
#     ],
#     'responses': {
#         200: {
#             'description': 'Xu hướng tìm kiếm và truy vấn liên quan giả lập'
#         }
#     }
# })
# def search_trends():
#     keyword = request.args.get("q")
#     geo = request.args.get("geo", "VN")

#     if not keyword:
#         return jsonify({"error": "Thiếu từ khóa (q)"}), 400

#     trends_data = get_simulated_search_trends(keyword, geo)
#     return jsonify(trends_data)


# if __name__ == "__main__":
#     # Hugging Face Spaces sẽ cung cấp PORT qua biến môi trường
#     # Mặc định của Flask là 5000, chúng ta sẽ dùng nó nếu PORT không được cung cấp.
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=False)
# import os
# import requests # Để tải file từ URL
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np
# import pickle
# from flask import Flask, request, jsonify
# from flasgger import Swagger, swag_from
# from flask_limiter import Limiter
# from flask_limiter.util import get_remote_address

# # Import các hàm từ utils.py (đảm bảo utils.py cũng nằm trong thư mục 'api/')
# from .utils import ( # Dùng import tương đối
#     stable_randint,
#     extract_keywords_tfidf,
#     analyze_sections,
#     get_simulated_pagespeed_score,
#     get_simulated_top_keywords,
#     get_simulated_keyword_rank,
#     get_simulated_search_trends,
#     preprocess_text
# )

# # --- Cấu hình tải mô hình từ Firebase Storage ---
# MODEL_URL = os.getenv("MODEL_URL") # URL công khai của mô hình trên Firebase Storage
# TOKENIZER_URL = os.getenv("TOKENIZER_URL") # URL công khai của tokenizer
# MODEL_PATH = "/tmp/keyword_model.h5" # Đường dẫn tạm thời trong serverless function
# TOKENIZER_PATH = "/tmp/Tokenizer.pkl"

# model = None
# tokenizer = None
# MAX_SEQUENCE_LENGTH = 200 # Phải khớp với khi huấn luyện mô hình

# def load_model_from_cloud():
#     global model, tokenizer
#     if model is not None and tokenizer is not None:
#         return

#     print("Đang tải mô hình và tokenizer từ Firebase Storage...")
#     try:
#         # Tạo thư mục /tmp nếu chưa có (thường có sẵn trong serverless env)
#         os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

#         # Tải mô hình
#         with requests.get(MODEL_URL, stream=True) as r:
#             r.raise_for_status()
#             with open(MODEL_PATH, 'wb') as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     f.write(chunk)
#         print(f"Đã tải mô hình xuống: {MODEL_PATH}")

#         # Tải tokenizer
#         with requests.get(TOKENIZER_URL, stream=True) as r:
#             r.raise_for_status()
#             with open(TOKENIZER_PATH, 'wb') as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     f.write(chunk)
#         print(f"Đã tải tokenizer xuống: {TOKENIZER_PATH}")

#         model = tf.keras.models.load_model(MODEL_PATH)
#         with open(TOKENIZER_PATH, "rb") as f:
#             tokenizer = pickle.load(f)
#         print("Mô hình và tokenizer đã được tải và nạp thành công.")

#     except Exception as e:
#         print(f"Lỗi khi tải hoặc nạp mô hình/tokenizer từ Firebase Storage: {e}")
#         model = None
#         tokenizer = None


# # Khởi tạo app Flask
# app = Flask(__name__)
# Swagger(app) # Khởi tạo Swagger

# limiter = Limiter(
#     get_remote_address,
#     app=app,
#     default_limits=["30 per minute", "1000 per day"]
# )

# # Đảm bảo các route gọi load_model_from_cloud()
# @app.route("/predict", methods=["POST"])
# @limiter.limit("10 per minute")
# @swag_from({ # ... Swagger docs ...
# })
# def predict():
#     load_model_from_cloud() # Đảm bảo mô hình được tải
#     if model is None or tokenizer is None:
#         return jsonify({"error": "Mô hình hoặc tokenizer chưa sẵn sàng. Vui lòng thử lại sau giây lát."}), 503
#     # ... (phần còn lại của predict) ...

# @app.route("/analyze", methods=["POST"])
# @limiter.limit("5 per minute")
# @swag_from({ # ... Swagger docs ...
# })
# def analyze():
#     load_model_from_cloud() # Đảm bảo mô hình được tải
#     if model is None or tokenizer is None:
#         return jsonify({"error": "Mô hình hoặc tokenizer chưa sẵn sàng. Vui lòng thử lại sau giây lát."}), 503
#     # ... (phần còn lại của analyze) ...

# # Không cần khối if __name__ == "__main__": app.run(...) khi deploy serverless

import os
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from flask import Flask, request, jsonify
# LOẠI BỎ: from flasgger import Swagger, swag_from
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Import các hàm từ utils.py (đảm bảo utils.py cũng nằm trong thư mục 'api/')
from .utils import (
    stable_randint,
    # LOẠI BỎ: extract_keywords_tfidf,
    analyze_sections,
    get_simulated_pagespeed_score,
    get_simulated_top_keywords,
    get_simulated_keyword_rank,
    get_simulated_search_trends,
    preprocess_text
)

# --- Cấu hình tải mô hình từ Firebase Storage ---
MODEL_URL = os.getenv("MODEL_URL")
TOKENIZER_URL = os.getenv("TOKENIZER_URL")
MODEL_PATH = "/tmp/keyword_model.h5"
TOKENIZER_PATH = "/tmp/Tokenizer.pkl"

model = None
tokenizer = None
MAX_SEQUENCE_LENGTH = 200

def load_model_from_cloud():
    global model, tokenizer
    if model is not None and tokenizer is not None:
        print("Mô hình đã được nạp, bỏ qua tải lại.")
        return

    print("Đang tải mô hình và tokenizer từ Firebase Storage...")
    if not MODEL_URL or not TOKENIZER_URL:
        print("Lỗi: MODEL_URL hoặc TOKENIZER_URL không được thiết lập trong biến môi trường.")
        return

    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Đã tải mô hình xuống: {MODEL_PATH}")

        with requests.get(TOKENIZER_URL, stream=True) as r:
            r.raise_for_status()
            with open(TOKENIZER_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Đã tải tokenizer xuống: {TOKENIZER_PATH}")

        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        print("Mô hình và tokenizer đã được tải và nạp thành công.")

    except Exception as e:
        print(f"Lỗi khi tải hoặc nạp mô hình: {e}")
        model = None
        tokenizer = None

app = Flask(__name__)
# LOẠI BỎ: Swagger(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["30 per minute", "1000 per day"]
)

# LOẠI BỎ @swag_from(...) decorators khỏi tất cả các routes

@app.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")
def predict():
    load_model_from_cloud()
    if model is None or tokenizer is None:
        return jsonify({"error": "Mô hình chưa sẵn sàng. Vui lòng kiểm tra log hoặc thử lại."}), 503

    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"prediction": 0, "confidence": 0.0, "message": "Không có văn bản để dự đoán"}), 400

    processed_text = preprocess_text(text)
    if not processed_text:
        return jsonify({"prediction": 0, "confidence": 0.0, "message": "Văn bản quá ngắn hoặc không có nội dung sau khi tiền xử lý."}), 200

    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    pred = model.predict(padded)[0][0]

    is_seo_friendly = int(pred > 0.5)
    confidence = float(pred)

    return jsonify({
        "prediction": is_seo_friendly,
        "confidence": confidence,
        "message": "Thân thiện với SEO" if is_seo_friendly else "Chưa thân thiện với SEO",
        "raw_prediction_score": confidence
    })


@app.route("/analyze", methods=["POST"])
@limiter.limit("5 per minute")
def analyze():
    load_model_from_cloud()
    if model is None or tokenizer is None:
        return jsonify({"error": "Mô hình chưa sẵn sàng. Vui lòng kiểm tra log hoặc thử lại."}), 503

    data = request.get_json()
    url = data.get("url", "")
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        full_text = soup.get_text(separator=' ', strip=True)
        if len(full_text) < 100:
            raise Exception("Không đủ nội dung để phân tích (ít hơn 100 ký tự).")

        processed_text_for_model = preprocess_text(full_text)

        web_score = 0
        content_score = 0
        if model and tokenizer and processed_text_for_model:
            sequence = tokenizer.texts_to_sequences([processed_text_for_model])
            padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
            pred = model.predict(padded)[0][0]
            web_score = int(pred * 100)
            content_score = int(min(95, web_score + stable_randint(-5, 5, url + "content")))
        else:
            web_score = stable_randint(40, 80, url + "web_score")
            content_score = stable_randint(50, 90, url + "content_score")

        keywords_top10 = int((web_score / 100 * 20) + stable_randint(5, 10, url + "kw_top10"))
        backlinks = int(500 + web_score * 50)

        # LOẠI BỎ keywords_extracted từ response
        # keywords_extracted = extract_keywords_tfidf(full_text) # LOẠI BỎ DÒNG NÀY

        section_analysis = analyze_sections(soup)
        pagespeed = get_simulated_pagespeed_score(url)

        overall_score = (
            (web_score * 0.4) +
            (pagespeed * 0.2) +
            (section_analysis["title_score"] * 0.15) +
            (section_analysis["h1_score"] * 0.15) +
            (section_analysis["meta_description_score"] * 0.05) +
            (section_analysis["https_score"] * 0.05)
        ) / 100
        overall_score = int(min(100, max(0, overall_score)))

        return jsonify({
            "url": url,
            "overall_seo_score": overall_score,
            "content_quality_score": web_score,
            "content_readability_score": content_score,
            "estimated_keywords_top10": keywords_top10,
            "estimated_backlinks": backlinks,
            # LOẠI BỎ: "extracted_keywords": keywords_extracted,
            "onpage_analysis": {
                "title": section_analysis["title"],
                "title_score": section_analysis["title_score"],
                "h1_tags": section_analysis["h1_tags"],
                "h1_score": section_analysis["h1_score"],
                "meta_description": section_analysis["meta_description"],
                "meta_description_score": section_analysis["meta_description_score"],
                "is_https": section_analysis["is_https"],
                "https_score": section_analysis["https_score"],
            },
            "pagespeed_score": pagespeed,
            "summary": f"Trang web {url} có điểm SEO tổng thể {overall_score}/100. Điểm chất lượng nội dung {web_score}/100, tốc độ {pagespeed}/100."
        })

    except requests.exceptions.RequestException as req_err:
        return jsonify({
            "error": f"Lỗi khi truy cập URL: {req_err}",
            "summary": f"Không thể kết nối hoặc tải nội dung từ URL: {url}"
        }), 400
    except Exception as e:
        return jsonify({
            "error": str(e),
            "summary": f"Lỗi khi phân tích URL: {url}"
        }), 400

# Các routes khác giữ nguyên nếu bạn vẫn cần chúng:
# /top_keywords, /keyword_rank, /search_trends
# Nhớ loại bỏ @swag_from(...) decorators khỏi chúng
if __name__ == "__main__":
    # Trong Docker, app thường chạy trên cổng cố định 5000 hoặc 8080
    # VÀ Dockerfile sẽ map cổng này ra ngoài.
    # Hugging Face Spaces sẽ tự xử lý việc ánh xạ cổng.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)