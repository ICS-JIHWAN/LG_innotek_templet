import os
import cv2
import glob
import numpy as np

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy.io import savemat
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]


def image_to_mat(root_path):
    """
    summary : Data 폴더를 읽은 후 image 파일을 .mat 확장자로 변경 해주는 함수.
              .mat 확장자 파일에는 정보 (Color image, Gray image, 경로, label, etc..)를 담고 있음.
              Data 폴더 안에 있는 폴더들을 label 로 생각.
    :param root_path: Root path
    :return:
    """
    # class 정보
    try:
        class_names = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    except:
        raise NotImplementedError('Checking root directory "{}!!"'.format(root_path))

    if len(class_names) != 0:
        # label encoding
        le  = LabelEncoder()
        int_label = le.fit_transform(class_names)

        # one hot encoding
        ohe = OneHotEncoder(sparse=False)
        one_hot_label = ohe.fit_transform(int_label.reshape(len(int_label), 1))

        print("데이터 변환 시작...")
        for cls_idx, cls in enumerate(class_names):
            data_paths = glob.glob(os.path.join(root_path, cls + '/*'))

            pbar = tqdm(enumerate(data_paths))
            for idx, path in pbar:
                p = Path(path)

                # 변환된 파일이 있으면 제외
                if (p.parent / f'{p.stem}.mat').exists():
                    continue

                try:
                    im = cv2.imread(path)
                    assert im is not None, f"opencv cannot read image correctly or {path} not exists"
                except:
                    im = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)
                    assert im is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"

                # .mat로 저장
                mat_dict = dict()
                mat_dict['path']  = path
                mat_dict['image'] = im
                mat_dict['integer_label'] = int_label[cls_idx]
                mat_dict['one_hot_label'] = one_hot_label[cls_idx]
                mat_dict['class'] = cls

                savemat(os.path.join(p.parent / f'{p.stem}.mat'), mat_dict)

        return le, ohe
    else:
        raise NotImplementedError('There is no class directory in the data folder!!')


if __name__ == "__main__":
    ROOT_PATH = "../data"

    os.listdir(ROOT_PATH)

    image_to_mat(ROOT_PATH)
