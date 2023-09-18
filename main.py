from data_preprocessing import load_eeg_data, preprocess_data, split_data
from model_definition import EEGNet
from train_eval import train_model
from test import test_model
import numpy as np


def main():
    # filename = f"subject_1.set"
    # # preprocess
    # data, labels = preprocess_data(load_eeg_data(filename))
    # X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, labels)

    # Load preprocessed data from .npy files
    X_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    X_val = np.load("x_val.npy")
    y_val = np.load("y_val.npy")
    X_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    # load train / test dataset
    num_channels = X_train.shape[1]  # 데이터에 따라 변경
    model = EEGNet(num_classes=len(np.unique(y_train)), num_channels=num_channels)

    train_model(model, X_train, y_train, X_val, y_val, 20)

    # test
    test_model(model, X_test, y_test)


if __name__ == "__main__":
    main()

# python main.py --train_data_file "230714_gd_jang_aiImage_1.vhdr" --infer_data_file "230714_gd_jang_aiImage_2" --content_type "video"
# 230714_gd_jang_aiImage_1, 230714_gd_jang_aiImage_2, 230714_gd_jang_Image_1, 230714_gd_jang_Image_2, 230814_js_kim_aiImage_1
# gd_jang, js_kim, kc_jeong, hs_oh
