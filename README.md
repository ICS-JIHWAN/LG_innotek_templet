# LG-Innotek Pytorch 기반 Sample Baseline Code

## Setup
### 1. Develop environments
* Window 10
* Python 3.8.0
* pypi(pip) installing 24.2

### 2. torch 2.4.0

* PC 환경에 맞게 설치 !
* CPU 환경
```
$ pip3 install torch torchvision torchaudio
```
* GPU 환경
  + [참고 블로그](https://foss4g.tistory.com/1565)
  + [PyTorch 공식 홈페이지](https://pytorch.org/get-started/locally/)
  + [CUDA Zone](https://developer.nvidia.com/cuda-zone)

### 3. Tensorboard
```
$ pip install tensorboard
```
* To upgrade past the version
```
$ pip uninstall tensorflow-tensorboard
$ pip install tensorboard
```

### 4. tqdm
```
$ pip install tqdm
```

### 5. scipy
* No module named scipy 발생 시 설치
```
$ pip install scipy
```

### 6. Open-CV 4.10.0
```
$ pip install opencv-python
```
* 버전확인
```
import cv2
print(cv2.__version__)
```

### 7. sklearn
```
$ pip install -U scikit-learn
```

### 8. PyYaml
```
$ pip install PyYAML
```