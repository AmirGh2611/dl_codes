from keras.models import Sequential
from keras.layers import Dense, Input

model = Sequential([
    Input(shape=(25,)),
    Dense(32, activation="relu"),
    Dense(64, activation="relu"),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax"),
])