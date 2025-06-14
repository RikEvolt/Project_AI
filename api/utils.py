import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import spacy
from spacy.lang.en import English
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Load spacy model (chỉ một lần)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model 'en_core_web_sm' (run 'python -m spacy download en_core_web_sm' manually if issues persist)")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Khởi tạo NLTK components (chỉ một lần)
# Cần tải stopwords nếu chưa có
try:
    stop_words_nltk = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    stop_words_nltk = set(stopwords.words('english'))

porter_stemmer = PorterStemmer()
punctuations = string.punctuation

# Hàm tạo số ngẫu nhiên "ổn định" dựa trên khóa
def stable_randint(min_val, max_val, key):
    # Sử dụng hash của key để tạo seed cố định
    np.random.seed(abs(hash(key)) % (2**32))
    return np.random.randint(min_val, max_val)

# Hàm chuẩn bị văn bản cho NLP
def preprocess_text(text):
    # Loại bỏ HTML tags (nếu input không phải từ BeautifulSoup)
    # Nếu input đã là text từ soup.get_text(), có thể bỏ qua bước này
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text(separator=' ', strip=True)

    # Chuyển về chữ thường
    clean_text = clean_text.lower()

    # Loại bỏ dấu câu và số (có thể giữ số nếu cần phân tích đặc thù)
    clean_text = re.sub(r'[^a-z\s]', '', clean_text) # chỉ giữ chữ cái và khoảng trắng

    # Tokenization, loại bỏ stop words và stemming/lemmatization
    doc = nlp(clean_text)
    # Sử dụng spaCy cho lemmatization và loại bỏ stop words hiệu quả hơn
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

    # Nếu muốn dùng NLTK stemmer
    # tokens = [porter_stemmer.stem(word) for word in tokens]

    return " ".join(tokens)


# Hàm trích xuất từ khóa bằng TF-IDF
def extract_keywords_tfidf(text, top_n=10):
    processed_text = preprocess_text(text) # Tiền xử lý văn bản
    if not processed_text:
        return []

    # Sử dụng TfidfVectorizer từ scikit-learn
    # Không cần stop_words='english' ở đây vì đã xử lý bằng spaCy/NLTK
    vectorizer = TfidfVectorizer(max_features=1000)
    try:
        X = vectorizer.fit_transform([processed_text])
        scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_scores[:top_n]]
    except ValueError: # Xử lý trường hợp văn bản rỗng sau tiền xử lý
        return []


# Hàm phân tích các phần chính của trang web (Title, H1)
def analyze_sections(soup):
    result = {}
    
    # Title
    title = soup.title.text.strip() if soup.title else ""
    result["title"] = title
    result["title_score"] = 100 if 30 <= len(title) <= 65 else 60 # Điểm dựa trên độ dài tiêu đề

    # H1 tags
    h1_tags = soup.find_all("h1")
    result["h1_tags"] = [h.get_text(strip=True) for h in h1_tags]
    result["h1_score"] = 100 if len(h1_tags) == 1 else (50 if len(h1_tags) > 1 else 0) # Điểm dựa trên số lượng H1 (nên có 1)

    # Meta Description (Thêm vào để đầy đủ hơn)
    meta_description_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_description_tag["content"].strip() if meta_description_tag and "content" in meta_description_tag.attrs else ""
    result["meta_description"] = meta_description
    # Điểm dựa trên độ dài meta description (thường 120-160 ký tự là tốt)
    result["meta_description_score"] = 100 if 120 <= len(meta_description) <= 160 else 60

    # Kiểm tra HTTPS
    is_https = False
    if soup.base and soup.base.get('href'):
        if 'https://' in soup.base.get('href', ''):
            is_https = True
    
    # Kiểm tra thêm nếu URL của request là HTTPS
    # requests.utils.urlparse(soup.request.url) chỉ khả dụng nếu response object được gán vào soup
    # Giả sử bạn truyền response vào hàm analyze_sections
    # Nếu không, ta chỉ kiểm tra base href hoặc bỏ qua phần này
    
    # CÁCH ĐƠN GIẢN HÓA VÀ AN TOÀN HƠN:
    # Bạn nên truyền trực tiếp cái URL gốc vào đây, hoặc response object
    # Giả sử trong analyze() bạn truyền response.url vào đây.
    # Để đơn giản hóa, tôi sẽ sử dụng URL mà BeautifulSoup đã tải.
    
    # Lấy URL cuối cùng mà requests tải về, vì có thể có redirect
    final_url = soup.request.url if hasattr(soup.request, 'url') else ''
    
    if final_url.startswith('https://'):
        result["is_https"] = True
    else:
        result["is_https"] = False

    # Thêm kiểm tra từ base href nếu có
    if soup.base and soup.base.get('href', '').startswith('https://'):
        result["is_https"] = True # Ưu tiên HTTPS nếu base href là HTTPS

    result["https_score"] = 100 if result["is_https"] else 50 # HTTPS là yếu tố xếp hạng

    return result

# Hàm giả lập Pagespeed Score (vì không dùng API)
# Thay vì gọi Google PageSpeed API, chúng ta sẽ giả lập một giá trị.
# Trong một hệ thống thực tế, bạn sẽ cần một công cụ kiểm tra tốc độ riêng.
def get_simulated_pagespeed_score(url):
    # Một cách đơn giản để tạo ra một điểm số "có vẻ thực"
    # Dựa trên độ dài URL hoặc một hash của URL để tạo sự nhất quán cho cùng một URL
    base_score = 70 # Điểm cơ bản
    # Tạo một chút ngẫu nhiên nhưng ổn định
    random_factor = stable_randint(-10, 20, url)
    score = min(100, max(30, base_score + random_factor))
    return score

# Hàm giả lập tìm kiếm top keyword và xếp hạng từ khóa
# Đây là phần khó nhất không dùng API.
# Chúng ta sẽ GIẢ LẬP dữ liệu dựa trên sự phổ biến nội bộ hoặc một danh sách từ khóa cố định.

# Danh sách từ khóa thịnh hành giả lập
SIMULATED_TRENDING_KEYWORDS = {
    "VN": [
        "thị trường bất động sản", "ai tiếng việt", "xu hướng công nghệ 2024",
        "du lịch việt nam", "giá vàng hôm nay", "chứng khoán phái sinh",
        "điện thoại mới nhất", "sức khỏe cộng đồng", "lập trình python",
        "học máy ứng dụng", "chuyển đổi số doanh nghiệp", "an ninh mạng",
        "năng lượng tái tạo", "thực phẩm hữu cơ", "ô tô điện",
        "thời trang bền vững", "giáo dục trực tuyến", "blockchain",
        "tín dụng đen", "biến đổi khí hậu", "tiền điện tử",
        "ứng dụng di động", "kinh tế số", "phát triển bền vững",
        "trí tuệ nhân tạo", "robot tự động", "công nghệ sinh học",
        "thành phố thông minh", "nông nghiệp công nghệ cao", "y tế từ xa",
        "thị trường lao động", "thực trạng giáo dục", "văn hóa đọc",
        "phim chiếu rạp", "âm nhạc việt", "sách hay",
        "khởi nghiệp trẻ", "quản trị kinh doanh", "tiếp thị số",
        "phát triển cá nhân", "tài chính cá nhân", "đầu tư thông minh",
        "nghệ thuật sống", "ẩm thực việt", "du lịch xanh",
        "sản phẩm thân thiện môi trường", "xe máy điện", "mạng 5g",
        "khoa học dữ liệu", "thiết kế đồ họa", "content marketing"
    ],
    "US": [
        "AI advancements", "election news", "sustainable living",
        "tech startups", "inflation rates", "stock market trends",
        "new phone releases", "mental health awareness", "learn python",
        "machine learning applications", "digital transformation", "cybersecurity threats",
        "renewable energy solutions", "organic food benefits", "electric vehicles",
        "sustainable fashion", "online education platforms", "cryptocurrency news",
        "gig economy trends", "climate change impact", "virtual reality",
        "mobile app development", "digital economy", "sustainable development goals",
        "artificial intelligence research", "robotics innovation", "biotechnology breakthroughs",
        "smart city initiatives", "agritech innovations", "telemedicine services",
        "job market forecast", "education reform", "reading culture",
        "movie releases", "popular music", "best books",
        "startup advice", "business management strategies", "digital marketing tips",
        "personal development hacks", "personal finance planning", "smart investing",
        "mindfulness practices", "healthy recipes", "eco-tourism destinations",
        "eco-friendly products", "electric scooters", "5g technology",
        "data science careers", "graphic design trends", "content marketing strategies"
    ]
}

def get_simulated_top_keywords(geo="VN", count=50):
    # Trả về danh sách từ khóa giả lập dựa trên khu vực
    return SIMULATED_TRENDING_KEYWORDS.get(geo.upper(), SIMULATED_TRENDING_KEYWORDS["VN"])[:count]

def get_simulated_keyword_rank(keyword, target_domain, max_pages=5):
    # Đây là hàm cực kỳ khó để giả lập một cách thuyết phục mà không có dữ liệu thật.
    # Chúng ta sẽ tạo một rank "ổn định" nhưng ngẫu nhiên.
    # Một cách đơn giản: nếu domain có tên keyword, hoặc một phần của keyword, rank sẽ tốt hơn.
    # Sử dụng stable_randint để đảm bảo tính nhất quán cho cùng một cặp keyword/domain

    keyword_lower = keyword.lower()
    domain_lower = target_domain.lower()

    if keyword_lower in domain_lower or any(part in domain_lower for part in keyword_lower.split()):
        # Nếu từ khóa hoặc một phần của từ khóa có trong domain, giả sử rank tốt hơn
        rank = stable_randint(1, 20, keyword + target_domain)
    else:
        # Nếu không, rank sẽ xấu hơn hoặc không tìm thấy
        rank = stable_randint(20, 100, keyword + target_domain)
        if rank > (max_pages * 10): # Giả sử không tìm thấy nếu rank quá lớn
            rank = -1
    return rank

# Hàm giả lập xu hướng tìm kiếm và truy vấn liên quan
def get_simulated_search_trends(keyword, geo="VN"):
    # Tạo dữ liệu xu hướng giả lập
    interest_data = []
    current_year = 2025 # Lấy năm hiện tại (từ thời gian bạn đưa ra)
    
    # Tạo dữ liệu cho 12 tháng gần nhất
    for i in range(12):
        month = (5 - i + 12) % 12 + 1 # Tháng hiện tại là 6 (tháng 6), quay ngược lại 12 tháng
        year = current_year if (month <= 6) else current_year -1 # Điều chỉnh năm
        
        # Giá trị xu hướng ngẫu nhiên ổn định theo từ khóa và tháng
        # Sử dụng hash của keyword + geo + month để tạo seed ổn định
        seed_key = f"{keyword}-{geo}-{year}-{month}"
        trend_value = stable_randint(30, 90, seed_key)
        
        interest_data.append({
            "date": f"{year}-{month:02d}-01", # Định dạng YYYY-MM-DD
            "value": trend_value
        })
    interest_data.reverse() # Sắp xếp lại theo thứ tự thời gian tăng dần

    # Tạo truy vấn liên quan giả lập
    related_queries = []
    base_related = [
        f"{keyword} là gì", f"cách {keyword}", f"{keyword} 2025",
        f"tìm hiểu về {keyword}", f"hướng dẫn {keyword}"
    ]
    # Thêm một vài từ khóa giả lập ngẫu nhiên
    for i in range(stable_randint(3, 7, keyword + "related")):
        related_keywords = get_simulated_top_keywords(geo, count=10)
        if related_keywords:
            random_related = related_keywords[stable_randint(0, len(related_keywords) - 1, keyword + f"rand{i}")]
            related_queries.append({
                "query": random_related,
                "value": stable_randint(50, 100, keyword + random_related)
            })
    
    return {
        "interest_over_time": interest_data,
        "related_queries": [{"query": q, "value": stable_randint(60, 100, keyword + q)} for q in base_related] + related_queries
    }