# pytorch.onnx.torch2trt
## pytorch.onnx.torch2trt

### resnet50

### 1.run
```python
python train.py # cfg.fp16=False|True
```

### 2.onnx
```python
python pytorch_to_onnx.py # cfg.fp16=False|True
```

### 3.torch2trt
```python
python pytorch_to_torch2trt.py
```
#### result
```python
Pytorch fp32 inference take 0.00661144495010376
Pytorch fp16 inference take 0.005471038818359375
Tensorrt fp32 inference 0.0007228851318359375
Tensorrt fp16 inference 0.0005936622619628906
```

### reference
1.[https://github.com/NVIDIA-AI-IOT/torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) <br/>
2.[https://github.com/kentaroy47/pytorch-onnx-tensorrt-CIFAR10](https://github.com/kentaroy47/pytorch-onnx-tensorrt-CIFAR10)

