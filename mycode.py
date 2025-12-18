import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# data normalizing
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
# one hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define the number of folds
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Store the results
fold_no = 1
train_scores = []
val_scores = []
test_scores = []

# K-fold Cross Validation
for train_idx, val_idx in kfold.split(x_train, y_train):
    print(f'\nTraining fold {fold_no}/{k_folds}...')

    # Split data for this fold
    x_train_fold = x_train[train_idx]
    y_train_fold = y_train[train_idx]
    x_val_fold = x_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Create model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(32, activation="relu"),
        Dense(64, activation="relu"),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ])

    model.compile(loss="categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    # Train model for this fold
    history = model.fit(x_train_fold, y_train_fold,
                        epochs=15,
                        batch_size=256,
                        validation_data=(x_val_fold, y_val_fold),
                        verbose=0)

    # Evaluate on validation set
    val_results = model.evaluate(x_val_fold, y_val_fold, verbose=0)
    val_scores.append(val_results[1])

    # Evaluate on test set
    test_results = model.evaluate(x_test, y_test, verbose=0)
    test_scores.append(test_results[1])

    # Plot training history for this fold
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f'Fold {fold_no} - Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f'Fold {fold_no} - Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.tight_layout()
    plt.show()

    fold_no += 1

# Print cross-validation results
print('\n' + '=' * 60)
print('CROSS-VALIDATION RESULTS')
print('=' * 60)

for i in range(len(val_scores)):
    print(f'Fold {i + 1}: Validation Accuracy = {val_scores[i]:.4f}, Test Accuracy = {test_scores[i]:.4f}')

print('\n' + '-' * 60)
print(f'Average Validation Accuracy: {np.mean(val_scores):.4f} (+/- {np.std(val_scores):.4f})')
print(f'Average Test Accuracy: {np.mean(test_scores):.4f} (+/- {np.std(test_scores):.4f})')
print('=' * 60)

# Optionally: Train a final model on all training data
print('\nTraining final model on entire training set...')

final_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(32, activation="relu"),
    Dense(64, activation="relu"),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax"),
])

final_model.compile(loss="categorical_crossentropy",
                    optimizer="rmsprop",
                    metrics=["accuracy"])

final_history = final_model.fit(x_train, y_train,
                                epochs=15,
                                batch_size=256,
                                validation_split=0.2,
                                verbose=1)

# Plot final model performance
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(final_history.history["loss"])
plt.plot(final_history.history["val_loss"])
plt.title('Final Model - Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(final_history.history["accuracy"])
plt.plot(final_history.history["val_accuracy"])
plt.title('Final Model - Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.show()

# Evaluate final model on test set
final_test_results = final_model.evaluate(x_test, y_test, verbose=0)
print(f'\nFinal Model Test Accuracy: {final_test_results[1]:.4f}')