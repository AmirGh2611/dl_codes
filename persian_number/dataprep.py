from dataset import load_hoda
from keras.utils import to_categorical

x_train, y_train, x_test, y_test = load_hoda()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
