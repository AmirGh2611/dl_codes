# for more information read "19-Intro2ML-HodaDataset.ipynb"
import cv2
import numpy as np
from scipy import io

def load_hoda(training_sample_size=1000, test_sample_size=200, size=5):
    #load dataset
    trs = training_sample_size
    tes = test_sample_size
    dataset = io.loadmat("Data_hoda_full.mat")

    #test and training set
    x_train_original = np.squeeze(dataset['Data'][:trs])
    y_train = np.squeeze(dataset['labels'][:trs])
    x_test_original = np.squeeze(dataset['Data'][trs:trs+tes])
    y_test = np.squeeze(dataset['labels'][trs:trs+tes])

    #resize
    x_train_5by5 = [cv2.resize(img, dsize=(size, size)) for img in x_train_original]
    x_test_5by_5 = [cv2.resize(img, dsize=(size, size)) for img in x_test_original]
    #reshape
    x_train = np.reshape(x_train_5by5, [-1,size**2])
    x_test = np.reshape(x_test_5by_5, [-1,size**2])
    
    return x_train, y_train, x_test, y_test