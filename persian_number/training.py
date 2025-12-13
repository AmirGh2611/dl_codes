from dataprep import x_train, y_train
from keras_model import model
import matplotlib.pyplot as plt

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=256,
                    validation_split=0.2)

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "validation"])
plt.show()
