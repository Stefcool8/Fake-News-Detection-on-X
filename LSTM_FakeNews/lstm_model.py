from keras.src.layers import LSTM, Dense, SpatialDropout1D, Embedding, Bidirectional, GlobalMaxPooling1D
from keras.src.models import Sequential
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf


class LSTMModel:
    def __init__(self, embedding_matrix, max_words=5000, max_len=50):
        self.embedding_matrix = embedding_matrix
        self.max_words = max_words
        self.max_len = max_len
        self.model = None

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.max_words, self.embedding_matrix.shape[1],
                                 weights=[self.embedding_matrix], trainable=False))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def hyperparameter_tuning(self, hp):
        model = Sequential()
        model.add(Embedding(self.max_words, self.embedding_matrix.shape[1],
                            weights=[self.embedding_matrix], input_length=self.max_len, trainable=False))
        model.add(SpatialDropout1D(0.2))
        model.add(Bidirectional(LSTM(hp.Int('units', min_value=64, max_value=256, step=64), return_sequences=True,
                                     dropout=hp.Float('dropout', 0.2, 0.5, step=0.1),
                                     recurrent_dropout=hp.Float('recurrent_dropout', 0.2, 0.5, step=0.1))))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self, x_train, y_train, x_val, y_val, batch_size=64, epochs=30):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

        history = self.model.fit(x_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(x_val, y_val),
                                 callbacks=[early_stopping, reduce_lr])
        return history

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def evaluate(self, x_val):
        y_pred_prob = self.model.predict(x_val)
        y_pred = (y_pred_prob > 0.5).astype(int)
        return y_pred, y_pred_prob
