import os
import cv2
import glob
import numpy as np
import pickle

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]


def image_to_pkl(root_path):
    """
    summary : Data 폴더를 읽은 후 image 파일을 .pkl 확장자로 변경 해주는 함수.
              .pkl 확장자 파일에는 정보 (Color image, 경로, label, etc..)를 담고 있음.
              Data 폴더 안에 있는 폴더들을 label 로 생각.
              pkl_class 형태의 폴더 안에 새로 저장
    :param root_path: Root path
    :return: number of classes
    """
    # class 정보
    try:
        class_names = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name)) and not('pkl' in name)]
    except:
        raise NotImplementedError('Checking root directory "{}!!"'.format(root_path))

    if len(class_names) != 0:
        num_class = len(class_names)
        # label encoding
        le  = LabelEncoder()
        int_label = le.fit_transform(class_names)

        # one hot encoding
        ohe = OneHotEncoder(sparse_output=False)
        one_hot_label = ohe.fit_transform(int_label.reshape(len(int_label), 1))

        print("데이터 변환 시작...")
        for cls_idx, cls in enumerate(class_names):
            # original data paths
            data_paths = glob.glob(os.path.join(root_path, cls + '/*'))

            # new directory 생성
            new_dir = os.path.join(root_path, 'pkl_' + cls)
            os.makedirs(new_dir, exist_ok=True)

            pbar = tqdm(enumerate(data_paths))
            for idx, path in pbar:
                p = Path(path)

                # 변환된 파일이 있으면 제외
                if (Path(new_dir) / f'{p.stem}.pkl').exists():
                    continue

                try:
                    im = cv2.imread(path)
                    assert im is not None, f"opencv cannot read image correctly or {path} not exists"
                except:
                    im = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)
                    assert im is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"

                # .pkl로 저장
                pkl_dict = dict()
                pkl_dict['original_path']  = path
                pkl_dict['image'] = im
                pkl_dict['integer_label'] = int_label[cls_idx]
                pkl_dict['one_hot_label'] = one_hot_label[cls_idx]
                pkl_dict['class'] = cls

                with open(Path(new_dir) / f'{p.stem}.pkl', 'wb') as f:
                    pickle.dump(pkl_dict, f)

        return le, num_class
    else:
        raise NotImplementedError('There is no class directory in the data folder!!')


if __name__ == "__main__":
    ROOT_PATH = "../data"

    os.listdir(ROOT_PATH)

    image_to_pkl(ROOT_PATH)
