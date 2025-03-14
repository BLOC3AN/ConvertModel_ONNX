<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# 

---

## Giáo Án Chi Tiết: Chuyển Đổi Mô Hình Keras sang ONNX và Triển Khai API Bảo Mật

### **Phần 1: Tổng Quan về ONNX và Lợi Ích**

**Lý do sử dụng ONNX**:

- **Tương thích đa framework**: ONNX giúp mô hình chạy trên TensorRT (NVIDIA), OpenVINO (Intel), CoreML (Apple), và các runtime khác[^1][^2].
- **Tối ưu hiệu suất**: Tăng tốc inference 2-3x nhờ tối ưu đồ thị tính toán và quantization[^4].
- **Triển khai linh hoạt**: Chạy trên đa nền tảng (Windows/Linux, CPU/GPU) và dễ tích hợp với hệ thống microservice[^3].

**Ứng dụng thực tế**:

- Triển khai YOLOv5 trên Raspberry Pi qua ONNX Runtime[^4].
- Chuyển đổi mô hình PyTorch sang TensorFlow Serving thông qua ONNX[^2].

---

### **Phần 2: Chuẩn Bị Môi Trường**

**Bước 1: Cài đặt thư viện**

```bash
# Cài tf2onnx và ONNX Runtime
pip install tf2onnx onnxruntime==1.15.1
```

**Bước 2: Chuẩn bị mô hình Keras**
Giả sử bạn có model đã lưu dưới dạng `model.keras`. Load lại mô hình:

```python
import tensorflow as tf
model = tf.keras.models.load_model('model.keras')
```

---

### **Phần 3: Chuyển Đổi Sang ONNX**

**Bước 1: Convert bằng Python API**

```python
import tf2onnx

onnx_model, _ = tf2onnx.convert.from_keras(
    model, 
    opset=15,  # Sử dụng opset 15 để hỗ trợ đầy đủ operator
    output_path="model.onnx"
)
```

**Bước 2: Kiểm tra tính hợp lệ**

```python
import onnx
onnx.checker.check_model(onnx_model)  # Kiểm tra lỗi cú pháp
```

**Giải thích tham số**:

- `opset`: Phiên bản operator set của ONNX (nên dùng >=13 để hỗ trợ LSTM/GRU).
- `output_path`: Đường dẫn lưu file .onnx.

---

### **Phần 4: Triển Khai Inference với ONNX Runtime**

**Bước 1: Load mô hình ONNX**

```python
import onnxruntime as ort

sess = ort.InferenceSession("model.onnx")
input_name = sess.get_inputs()[^0].name  # Ví dụ: 'input_1'
output_name = sess.get_outputs()[^0].name  # Ví dụ: 'output_1'
```

**Bước 2: Dự đoán**

```python
import numpy as np

# Giả lập dữ liệu đầu vào
dummy_input = np.random.randn(1, 28, 28).astype(np.float32)  # Chuẩn MNIST 28x28
result = sess.run([output_name], {input_name: dummy_input})
print("Predicted class:", np.argmax(result[^0]))
```

**Lưu ý**:

- Đảm bảo kiểu dữ liệu đầu vào (`float32`/`int64`) khớp với mô hình gốc.
- Sử dụng `onnxruntime-gpu` nếu chạy trên GPU NVIDIA[^1][^4].

---

### **Phần 5: Bảo Mật Khi Triển Khai API**

**1. Mã hóa mô hình**:

- **AES-256**: Mã hóa file `model.onnx` trước khi lưu trữ.

```python
from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher = Fernet(key)

with open("model.onnx", "rb") as f:
    encrypted_model = cipher.encrypt(f.read())

with open("model.encrypted", "wb") as f:
    f.write(encrypted_model)
```


**2. Bảo vệ API endpoint**:


| **Biện pháp** | **Triển khai** | **Ví dụ** |
| :-- | :-- | :-- |
| **Xác thực JWT** | Sử dụng FastAPI + OAuth2PasswordBearer | Kiểm tra token qua Auth0/Keycloak |
| **Rate Limiting** | Giới hạn 100 requests/phút | Dùng Redis hoặc NGINX |
| **Input Validation** | Kiểm tra kích thước ảnh <10MB | Dùng Pydantic BaseModel |

**3. Giám sát hệ thống**:

- **Prometheus + Grafana**: Theo dõi latency, memory usage, và error rate.
- **Audit Log**: Ghi lại IP, timestamp, và payload size của mọi request.

---

### **Phần 6: Tối Ưu Hóa Production**

**1. Quantization**:
Giảm kích thước mô hình bằng FP16:

```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QInt8  # Hoặc QuantType.QFLOAT16
)
```

**2. Docker Triển Khai**:

```Dockerfile
FROM python:3.9-slim
RUN pip install onnxruntime fastapi uvicorn cryptography
COPY model_quantized.onnx /app/
COPY main_api.py /app/
CMD ["uvicorn", "app.main_api:app", "--host", "0.0.0.0"]
```

---

### **Phần 7: Xử Lý Lỗi Thường Gặp**

| **Lỗi** | **Nguyên nhân** | **Cách khắc phục** |
| :-- | :-- | :-- |
| `TypeError: Input type not supported` | Kiểu dữ liệu đầu vào không khớp | Chuyển sang `np.float32` hoặc `np.int64` |
| `ONNXRuntimeError: Invalid Graph` | Operator không hỗ trợ trong opset | Nâng opset hoặc dùng `tf2onnx --opset 18` |

---

### **Tổng Kết**

- **Lợi ích ONNX**: Tăng tốc 3x inference, triển khai đa nền tảng, dễ tích hợp với TensorRT/OpenVINO.
- **Bảo mật**: Mã hóa mô hình + JWT + rate limiting để chống tấn công.
- **Tối ưu**: Quantization FP16/INT8 và triển khai qua Docker.

**Tài nguyên mở rộng**:

- [Hướng dẫn chính thức tf2onnx](https://github.com/onnx/tensorflow-onnx)
- [ONNX Runtime Inference Optimization](https://onnxruntime.ai/docs/performance/)

<div style="text-align: center">⁂</div>

[^1]: https://viblo.asia/p/onnx-va-tensorflow-3P0lP8wplox

[^2]: https://www.toolify.ai/vi/ai-news-vn/chuyn-i-m-hnh-ai-vi-pytorch-tensorflow-onnx-tensorrt-v-openvino-2503142

[^3]: https://atekco.io/1640060513523-thong-nhat-ky-thuat-trien-khai-mo-hinh-machine-learning-voi-onnx/

[^4]: https://docs.ultralytics.com/vi/integrations/onnx/

[^5]: https://viblo.asia/p/chuyen-doi-mo-hinh-hoc-sau-ve-onnx-bWrZnz4vZxw

[^6]: https://cntt.dlu.edu.vn/wp-content/uploads/2022/11/BaoCaoToanVan_Xay-dung-he-thong-diem-danh-sinh-vien-dua-tren-nhan-khuon-mat.pdf

[^7]: https://docs.ultralytics.com/vi/guides/steps-of-a-cv-project/

