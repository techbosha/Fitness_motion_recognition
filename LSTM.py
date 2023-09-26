import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers, optimizers
import loadcsv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


def create_model():
    model = Sequential()
    model.add(LSTM(256, return_sequences = True, input_shape = (30, 99)))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences = True))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences = False))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation = 'tanh'))
    model.add(Dense(13, activation = 'softmax'))
    adam = optimizers.Adam(learning_rate = 1e-3)
    model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
                  optimizer = adam, metrics = ['categorical_accuracy'])
    return model


def train(train_X, train_y, batch_size, epochs):
    model = create_model()
    model.summary()
    model_mckp = ModelCheckpoint('best_model_weights.h5',
                                 monitor = 'val_categorical_accuracy',
                                 save_best_only = True,
                                 save_weights_only = True,
                                 mode = 'max')

    early_stopping = EarlyStopping(monitor = 'val_categorical_accuracy',
                                   patience = 20,
                                   verbose = 0,
                                   restore_best_weights = True)

    history = model.fit(train_X, train_y,
                        epochs = epochs,
                        batch_size = batch_size,
                        validation_split = 0.25,
                        callbacks = [model_mckp, early_stopping])
    return model, history


def load(weight = 'best_model_weights.h5'):
    model = create_model()
    model.load_weights(weight)
    return model


if __name__ == "__main__":
    csv_file_path = 'csv_dataset_1'
    dir_label_df = loadcsv.get_csvdata(csv_file_path)
    train_data, test_data = train_test_split(dir_label_df, random_state = 777, train_size = 0.8)
    train_X = loadcsv.load_csv_from_csvpath(train_data)[1]
    test_X = loadcsv.load_csv_from_csvpath(test_data)[1]
    train_y = loadcsv.load_csv_from_csvpath(train_data)[0]
    test_y = loadcsv.load_csv_from_csvpath(test_data)[0]

    # hyperparameters
    epochs = 350
    batch_size = 60
    # train_X.shape(517, 30, 99)
    # train_y.shape(517, 22)
    n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]

    model, history = train(train_X = train_X, train_y = train_y,
                           batch_size = batch_size, epochs = epochs)
    
    #畫出預測模型的混淆矩陣結果
    y_pred = model.predict(test_X)
    y_pred_classes = []
    for item in y_pred:
        max_index = np.argmax(item)
        new_array = np.zeros_like(item)
        new_array[max_index] = 1
        y_pred_classes.append(new_array)
    y_pred_classes = np.array(y_pred_classes)
    cm = confusion_matrix(test_y.argmax(axis = 1), y_pred_classes.argmax(axis = 1))
    plt.figure(figsize = (10, 8))
    sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues",
                xticklabels = True, yticklabels = True)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    accuracy = model.evaluate(
        test_X, test_y, batch_size=len(test_X), verbose=1)
