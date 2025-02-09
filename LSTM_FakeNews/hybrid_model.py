import tensorflow as tf
from keras.src.layers import Conv1D, Dense, Embedding, Dropout, MaxPooling1D, GlobalMaxPooling1D, LSTM, \
    BatchNormalization, Bidirectional, SpatialDropout1D
from keras.src.models import Sequential
from keras.src.optimizers import Adam


class HybridModel:
    def __init__(self, embedding_matrix, max_words=5000, max_len=50):
        self.embedding_matrix = embedding_matrix
        self.max_words = max_words
        self.max_len = max_len
        self.model = None

    def build_model(self):
        self.model = Sequential()

        # Embedding layer
        self.model.add(Embedding(input_dim=self.max_words,
                                 output_dim=self.embedding_matrix.shape[1],
                                 weights=[self.embedding_matrix],
                                 trainable=False))

        self.model.add(SpatialDropout1D(0.2))
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.2))
        self.model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
        self.model.add(BatchNormalization())
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    def hyperparameter_tuning(self, hp):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_words,
                            output_dim=self.embedding_matrix.shape[1],
                            weights=[self.embedding_matrix],
                            input_length=self.max_len,
                            trainable=False))
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=hp.Int('filters', min_value=32, max_value=256, step=32),
                         kernel_size=hp.Int('kernel_size', min_value=3, max_value=5, step=1),
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(hp.Float('conv_dropout', 0.2, 0.5, step=0.1)))
        model.add(Bidirectional(LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32),
                                     return_sequences=True,
                                     dropout=hp.Float('lstm_dropout', 0.2, 0.5, step=0.1),
                                     recurrent_dropout=hp.Float('recurrent_dropout', 0.2, 0.5, step=0.1))))
        model.add(BatchNormalization())
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
            loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, x_train, y_train, x_val, y_val, batch_size=64, epochs=30):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

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
