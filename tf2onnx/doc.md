# ONNX - Open Neural Network Exchange 
> *Chuyển mô hình AI từ `tensorflow` sang `onnx`* [$^{[1]}$](https://github.com/onnx/tensorflow-onnx) 

- [ONNX - Open Neural Network Exchange](#onnx---open-neural-network-exchange)
- [Yêu cầu cấu hình](#yêu-cầu-cấu-hình)
    - [Python Virtual enviroment](#python-virtual-enviroment)
  - [Cài đặt thư viện](#cài-đặt-thư-viện)
- [Bắt đầu dự án](#bắt-đầu-dự-án)
  - [Setup workspace](#setup-workspace)
  - [Load model tensorflow](#load-model-tensorflow)
  - [Rebuild Model (Option)](#rebuild-model-option)
    - [Xây dựng model ban đầu](#xây-dựng-model-ban-đầu)
    - [Model đã build sẵn](#model-đã-build-sẵn)
  - [Chuyển sang ONNX](#chuyển-sang-onnx)
  - [Kiểm tra model ONNX](#kiểm-tra-model-onnx)
- [Test model ONNX](#test-model-onnx)
- [Tối ưu và Lượng tử hóa (Quantize)](#tối-ưu-và-lượng-tử-hóa-quantize)

# Yêu cầu cấu hình
>`ONNX` chỉ hỗ trợ Python `3.7-3.10`    
Nếu version Python >3.10 thì nên cài `virtual enviroment`


<details close>
<summary> Cài đặt virtual enviroment</summary>  

### Python Virtual enviroment
- Cài trực tiếp
```bash
python3 -m venv <Tên venv>
```
>Python3 hỗ trợ tạo virtual enviroment từ phiên bản `3.3`
Nếu muốn chỉ định phiên bản cụ thể ví dụ `Python 3.8` thì cài bằng `pyenv`
```bash
pyenv install 3.8.0
pyenv local 3.8.0 # Cài đặt cho dự án hiện tại
# pyenv global 3.8.0 # Cài đặt cho toàn cục
```

</details>


## Cài đặt thư viện 
- Cài thư viện `onnx`
```bash
pip install onnxruntime
pip install -U tf2onnx
```
- Cài thư viện tensorflow (nếu chưa có)    
**[Tensorflow có GPU](https://www.tensorflow.org/install/source#gpu) chỉ hỗ trợ cho phiên bản 2.18 trở về trước**
```bash
pip install tensorflow==2.18.0
```
>[Nên cài virtual enviroment](#python-virtual-enviroment) trước khi chạy tensorflow

# Bắt đầu dự án
## Setup workspace
```py
import os
os.chdir("~/convert_model_ONNX/")
```
## Load model tensorflow 
Nếu đã có model training trước đó
```py
import tensorflow as tf
path_model = "./models_hub/model.keras"
model = tf.keras.models.load_model(path_model)
```
## Rebuild Model (Option)
- Những model có kiến trúc `BiLSTM`, `LSTM`, `RNN`, `GRU` thưởng sử dụng `CUDNN` trong quá trình build và training nhưng `ONNX` không hỗ trợ `CUDNN`.   
- Thường sẽ gặp lỗi `CudnnRNNV3` trong quá trình chuyển sang `ONNX`
> ValidationError: No Op registered for CudnnRNNV3 with domain_version of 15  
==> Context: Bad node spec for node. Name: StatefulPartitionedCall/sequential_1/bidirectional_1/forward_lstm_1/CudnnRNNV3 OpType: CudnnRNNV3

&rarr; Ta sẽ thêm attribute `implementation=1` vào các lớp layer đó.  

### Xây dựng model ban đầu
Định nghĩa mô hình `BiLSTM` đơn giản với `implementation=1` trong các lớp BiLSTM
```python
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
# Định nghĩa lại mô hình với implementation=1
model = tf.keras.Sequential([
    Bidirectional(LSTM(64, return_sequences=True, implementation=1), input_shape=(None, 10)),  # Thay 10 bằng số đặc trưng thực tế
    Bidirectional(LSTM(32, implementation=1)),
    Dense(10, activation='softmax')
])
```
### Model đã build sẵn
Khi đã có model build sẵn, lúc này ta không thể buils lại model nữa mà ta chèn các `implementation=1` trược tiếp vào model đã build.
```py
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Bidirectional):
        # Truy cập tầng forward và backward
        forward_lstm = layer.forward_layer
        backward_lstm = layer.backward_layer
        # Kiểm tra và thay đổi implementation
        if hasattr(forward_lstm, 'implementation') and forward_lstm.implementation != 1:
            print(f"Thay đổi implementation của {forward_lstm.name} sang 1")
            forward_lstm.implementation = 1
        if hasattr(backward_lstm, 'implementation') and backward_lstm.implementation != 1:
            print(f"Thay đổi implementation của {backward_lstm.name} sang 1")
            backward_lstm.implementation = 1
# Xây dựng lại mô hình với input shape (thay đổi theo thực tế)
input_format = (None, 100, 20)
model.build(input_format) # Ví dụ: (batch, timesteps, features)
# Lưu dưới dạng SavedModel
model.export('./models_hub/model_tf2onnx/')
```
- `input_format`: Cần định nghĩa lại format đầu vào với bài toán LSTM thì format dạng `INPUT_SHAPE=(batch_size, timestep, features)`  
- Lưu model dưới dạng thư mục với `model.export('/path/dir/')`


## Chuyển sang ONNX
```py
onnx_model = './models_hub/model.onnx'
!python3 -m tf2onnx.convert --saved-model "./models_hub/model_tf2onnx/" --output {onnx_model} --opset 15
```
## Kiểm tra model ONNX
```py
import onnx
onnx_model = onnx.load(onnx_model)
onnx.checker.check_model(onnx_model)
print("Model ONNX hợp lệ!")
```
# Test model ONNX
```py
import onnxruntime as ort
import numpy as np
session = ort.InferenceSession('./models_hub/model.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
dummy_input = np.random.randn(1, 100, 20).astype(np.float32) 
outputs = session.run([output_name], {input_name: dummy_input})
```
# Tối ưu và Lượng tử hóa (Quantize)
> Document: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
```py
from onnxruntime.quantization import quantize_dynamic,QuantType
quantize_dynamic(
    model_input = "./models_hub/model.onnx",
    model_output="./models_hub/model_quantized.onnx",
    weight_type=QuantType.QInt8,
)
```


[def]: #tối-ưu-và-lượng-tử-hóa-quantize