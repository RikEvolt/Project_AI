# train_simple_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os

# Bước 1: Tạo dữ liệu giả lập
# Văn bản thân thiện SEO (label = 1)
seo_friendly_texts = [
    # "Learn SEO best practices for higher rankings on Google.",
    # "Keyword research is crucial for organic traffic.",
    # "Optimize your website content with relevant keywords.",
    # "Technical SEO ensures your site is crawlable and indexable.",
    # "Build high-quality backlinks to improve domain authority.",
    # "Mobile-first indexing impacts your search visibility.",
    # "Content marketing drives valuable organic leads.",
    # "User experience (UX) is a ranking factor in SEO.",
    # "Analyze Google Search Console for performance insights.",
    # "Core Web Vitals measure website speed and responsiveness."
    # Công nghệ & AI
    "Trí tuệ nhân tạo phát triển",
    "ứng dụng AI mới nhất 2025",
    "học máy cho người mới bắt đầu",
    "robot tự động hóa công nghiệp",
    "công nghệ sinh học đột phá",
    "xe điện xu hướng tương lai",
    "mạng 5G và tiềm năng",
    "khoa học dữ liệu và phân tích",
    "phát triển ứng dụng di động",
    "điện toán đám mây giải pháp doanh nghiệp",
    "thực tế ảo tăng cường VR AR",
    "chip bán dẫn thế hệ mới",
    "blockchain và tiền điện tử",
    "an ninh mạng chống tấn công mạng",
    "chuyển đổi số trong kinh doanh",
    "xu hướng công nghệ 2025",
    "tối ưu hóa hiệu suất website",
    "marketing kỹ thuật số hiệu quả",
    "chiến lược SEO toàn diện",
    "cập nhật thuật toán Google",

    # Sức khỏe & Y tế
    "cách tăng cường hệ miễn dịch",
    "chế độ ăn keto giảm cân",
    "tập thể dục tại nhà hiệu quả",
    "sức khỏe tinh thần và stress",
    "phòng chống dịch bệnh mới",
    "y học cổ truyền trị liệu",
    "dinh dưỡng cho người tiểu đường",
    "lợi ích của thiền định",
    "ngủ đủ giấc và năng suất",
    "khám sức khỏe định kỳ",

    # Kinh tế & Tài chính
    "thị trường chứng khoán hôm nay",
    "giá vàng thế giới",
    "đầu tư bất động sản 2025",
    "lãi suất ngân hàng mới nhất",
    "quản lý tài chính cá nhân",
    "kinh tế vĩ mô việt nam",
    "lạm phát và tác động",
    "vay tín chấp nhanh",
    "tiết kiệm thông minh",
    "phát triển bền vững kinh tế xanh",

    # Du lịch & Giải trí
    "địa điểm du lịch hè 2025",
    "review phim chiếu rạp mới nhất",
    "show truyền hình hot 2025",
    "khám phá ẩm thực đường phố",
    "sách hay nên đọc",
    "âm nhạc thịnh hành",
    "game online được yêu thích",
    "lễ hội văn hóa truyền thống",
    "mẹo du lịch tiết kiệm",
    "phượt miền núi phía bắc",

    # Xã hội & Đời sống
    "biến đổi khí hậu và môi trường",
    "giáo dục trực tuyến hiệu quả",
    "tình hình việc làm 2025",
    "kỹ năng mềm cho sinh viên",
    "lối sống tối giản",
    "phát triển cộng đồng bền vững",
    "quyền riêng tư dữ liệu cá nhân",
    "chính sách an sinh xã hội",
    "thiết kế nhà ở hiện đại",
    "nuôi dạy con cái thời 4.0",
    "hướng dẫn làm bánh",
    "mẹo vặt cuộc sống",
    "phong cách thời trang 2025",
    "các loại cây cảnh phong thủy",
    "cách nấu ăn ngon đơn giản",
    "làm đẹp tự nhiên tại nhà",
    "chăm sóc da mặt cơ bản",
    "tự học tiếng Anh giao tiếp",
    "yoga cho người mới bắt đầu",
    "thiết kế nội thất chung cư",

    # Từ khóa liên quan trực tiếp đến SEO (để mô hình hiểu ngữ cảnh)
    "nghiên cứu từ khóa hiệu quả",
    "xây dựng liên kết chất lượng",
    "tối ưu hóa tốc độ website",
    "on-page SEO checklist",
    "công cụ SEO miễn phí tốt nhất",
    "SEO Local cho doanh nghiệp nhỏ",
    "phân tích đối thủ cạnh tranh SEO",
    "viết bài chuẩn SEO",
    "audit SEO website",
    "SEO copywriting techniques"
]

# Văn bản không thân thiện SEO (label = 0)
non_seo_friendly_texts = [
    # "Random text without any specific focus.",
    # "Buy cheap shoes online for good prices.",
    # "Click here to see amazing deals now!!!",
    # "This is a very long sentence that talks about nothing important and lacks structure.",
    # "Spammy links can hurt your website ranking.",
    # "Just rambling words with no clear topic or purpose.",
    # "Hello world, programming is fun sometimes.",
    # "Financial advice consult a professional for your investments.",
    # "Cute cat videos trending on social media today.",
    # "Weather forecast for tomorrow sunny and warm."
    # Từ ngữ ngẫu nhiên/ít liên quan
    "cái bàn màu xanh cũ kỹ",
    "tiếng mèo kêu ban đêm",
    "lốp xe đạp thủng ở đâu",
    "cách sửa cái đèn hỏng",
    "đi dép lê trong nhà",
    "mùi hương của hoa sen",
    "nước chảy đá mòn là gì",
    "mặt trăng tròn vào thứ ba",
    "giày cao gót màu tím",
    "bút chì hết mực",

    # Từ khóa quá cụ thể/hẹp
    "giá ốc vít M3x10 ở cửa hàng số 7",
    "cách trồng cây cà chua trong chậu nhựa 5 lít vào mùa đông năm 2020",
    "lịch sử chiếc ghế của ông tôi",
    "hướng dẫn sửa máy giặt electrolux đời 2015 model ABCD1234 ở quận 1",
    "mã màu RAL 7016 trong thiết kế nội thất công nghiệp tại Hà Nội",
    "bài hát về con ếch màu xanh lam",
    "công thức nấu món súp bí đỏ kiểu Pháp cổ điển của đầu bếp Pierre",
    "kích thước tấm ván gỗ thông nhập khẩu từ Phần Lan loại AAA",
    "ý nghĩa tên gọi của cây xương rồng không gai ở sa mạc Sahara",
    "quán cà phê có tên 123 đường ABCD phường X ở thành phố Y",

    # Cụm từ không có nghĩa
    "chạy nhanh hơn cả gió biển",
    "mây trời bay lượn màu xanh",
    "tiếng chim hót líu lo trên cành cây khô",
    "sự im lặng của bóng tối",
    "cảm xúc của chiếc lá rơi",
    "vũ điệu của những hạt mưa",
    "ánh sáng xuyên qua kẽ lá",
    "hương vị của ký ức cũ",
    "tiếng lòng của đại dương sâu thẳm",
    "bí mật của ngôi sao lấp lánh",

    # Các câu hỏi/phát biểu không mang tính tìm kiếm rộng rãi
    "hôm nay tôi ăn gì",
    "bạn có khỏe không",
    "mấy giờ rồi",
    "tôi nên làm gì bây giờ",
    "thời tiết ngày mai thế nào",
    "có tin gì mới không",
    "kể cho tôi nghe một câu chuyện",
    "tôi muốn ngủ",
    "tôi không biết gì cả",
    "mệt quá đi thôi"
]

# Kết hợp dữ liệu
texts = seo_friendly_texts + non_seo_friendly_texts
labels = [1] * len(seo_friendly_texts) + [0] * len(non_seo_friendly_texts)

# Bước 2: Tiền xử lý văn bản và tạo Tokenizer
MAX_VOCAB_SIZE = 1000
MAX_SEQUENCE_LENGTH = 200 # Phải khớp với MAX_SEQUENCE_LENGTH trong app.py

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<unk>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Chuyển đổi labels sang numpy array
labels = np.array(labels)

# Bước 3: Chia dữ liệu huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Bước 4: Xây dựng mô hình TensorFlow/Keras đơn giản
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(MAX_VOCAB_SIZE, 16, input_length=MAX_SEQUENCE_LENGTH),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Sigmoid cho phân loại nhị phân
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Bước 5: Huấn luyện mô hình
print("\nBắt đầu huấn luyện mô hình...")
history = model.fit(
    X_train, y_train,
    epochs=10, # Số lượng epoch thấp cho ví dụ
    validation_data=(X_test, y_test),
    verbose=1
)
print("Huấn luyện hoàn tất.")

# Bước 6: Lưu mô hình và tokenizer
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "keyword_model.h5")
tokenizer_path = os.path.join(model_dir, "Tokenizer.pkl")

model.save(model_path)
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"\nMô hình đã lưu tại: {model_path}")
print(f"Tokenizer đã lưu tại: {tokenizer_path}")
print("Giờ bạn có thể chạy file app.py")