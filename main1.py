import re,os                                   # 're' Replication of text.
import numpy as np
import pandas as pd                         # 'pandas' to manipulate the dataset.
import matplotlib.pyplot as plt
import seaborn as sns                       # 'seaborn' to visualize the dataset.
import tensorflow as tf
from tensorflow.keras.models import Sequential                # 'Sequential' model will be used for training.
from sklearn.model_selection import train_test_split          # 'train_test_split' for splitting the data into train and test data.
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences       # 'pad_sequences' for having same dimmension for each sequence.
from tensorflow.keras.layers import Embedding, Bidirectional,LSTM, Flatten, Dense,Input,Average,Reshape,Dropout,Concatenate,Maximum     # import some layers for training.
from tensorflow.keras.layers import Conv2D, MaxPool2D,Convolution1D,MaxPooling1D     # import some layers for training.
from tensorflow.keras.utils import to_categorical
import gensim.models.keyedvectors as word2vec #need to use due to depreceated model
# import gensim.models.word2vec as Word2Vec #need to use due to depreceated model
import gc
from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import pandas as pd
from keras.utils import to_categorical

X_train = pd.read_excel(r"Data_train.xlsx")

if X_train.isnull().values.any():
    X_train = X_train.dropna()

print(X_train.shape)

X_train = X_train[['title', 'text', 'rating']]
X_train_title = X_train['title'].apply(str)
X_train_text = X_train['text'].apply(str)

y_train = X_train['rating']

import pandas as pd
from keras.utils import to_categorical

X_test = pd.read_excel(r"Data_test.xlsx")

if X_test.isnull().values.any():
    X_test = X_test.dropna()

print(X_test.shape)
print(X_test.columns)

X_test = X_test[['title', 'text', 'rating']]
X_test_title = X_test['title'].apply(str)
X_test_text = X_test['text'].apply(str)

y_test = X_test['rating']
test_labels = to_categorical(y_test - 1, num_classes=5)

w2vModel = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=50000)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train_title)
tokenizer.fit_on_texts(X_train_text)
tokenizer.fit_on_texts(X_test_title)
tokenizer.fit_on_texts(X_test_text)

train_title = tokenizer.texts_to_sequences(X_train_title)
train_text = tokenizer.texts_to_sequences(X_train_text)
test_title = tokenizer.texts_to_sequences(X_test_title)
test_text = tokenizer.texts_to_sequences(X_test_text)

vocab_size = len(tokenizer.word_index) + 1

max_len_title = 30
max_len_text = 150

train_title = pad_sequences(train_title, padding = 'post', maxlen = max_len_title)
train_text = pad_sequences(train_text, padding = 'post', maxlen = max_len_text)
test_title = pad_sequences(test_title , padding = 'post', maxlen = max_len_title)
test_text = pad_sequences(test_text , padding = 'post', maxlen = max_len_text)

# BiLSTM Titles+Contents
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.optimizers import Adam

filter_sizes = [3, 4, 5]
num_filters = 100
drop = 0.5
learning_rate = 0.01

embedding_dim = 50
max_words = w2vModel.vectors.shape[0]

def build_classifier_model_bilstm_title_content(learning_rate):
    # Titles
    inputs_title = Input(shape=(max_len_title,), dtype='int32', name='input_title')
    embedding_title = Embedding(max_words, embedding_dim,
                                weights=[w2vModel.vectors[:max_words, :embedding_dim]],
                                input_length=max_len_title,
                                trainable=True)(inputs_title)  # Set trainable to True
    lstm_title = Bidirectional(LSTM(256, activation='tanh', return_sequences=True))(embedding_title)

    # Text
    inputs_text = Input(shape=(max_len_text,), dtype='int32', name='input_text')
    embedding_text = Embedding(max_words, embedding_dim,
                               weights=[w2vModel.vectors[:max_words, :embedding_dim]],
                               input_length=max_len_text,
                               trainable=True)(inputs_text)  # Set trainable to True
    lstm_text = Bidirectional(LSTM(256, activation='tanh', return_sequences=True))(embedding_text)

    # Additional LSTM layer
    lstm_title_2 = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(lstm_title)
    lstm_text_2 = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(lstm_text)

    # Apply GlobalAveragePooling1D to the output sequences
    gap_title = GlobalAveragePooling1D()(lstm_title_2)
    gap_text = GlobalAveragePooling1D()(lstm_text_2)

    # Fully connected layers for title and text
    fc_title = Dense(128, activation='relu')(gap_title)
    fc_text = Dense(128, activation='relu')(gap_text)

    # Concatenate the outputs
    concatenated_output = Concatenate(axis=-1)([fc_title, fc_text])
    dropout_output = Dropout(drop)(concatenated_output)

    # Additional Dense layer
    fc_combined = Dense(64, activation='relu')(dropout_output)

    preds = Dense(3, activation='softmax', name='classifier')(fc_combined)

    # This creates a model that includes inputs and outputs
    model_BiLstm = tf.keras.Model(inputs=[inputs_title, inputs_text], outputs=preds)

    # Use the Adam optimizer with the specified learning rate
    optimizer = Adam(learning_rate=learning_rate)

    model_BiLstm.compile(loss='categorical_crossentropy',
                          optimizer=optimizer,
                          metrics=['acc'])
    return model_BiLstm

model_BiLstm = build_classifier_model_bilstm_title_content(learning_rate)
model_BiLstm.summary()

model_BiLstm.save_weights('BiLSTM.h5')
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow as tf

# Function for learning rate scheduling
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

learning_rate = 0.01
model_BiLstm = build_classifier_model_bilstm_title_content(learning_rate)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = LearningRateScheduler(lr_scheduler)

# Train the model with callbacks
history = model_BiLstm.fit(
    [train_title, train_text],
    train_labels,
    batch_size=4,
    epochs=100,
    validation_split=0.10,
    verbose=1,
    callbacks=[early_stopping, lr_schedule]
)

# Evaluate the model on the test set
score = model_BiLstm.evaluate([test_title, test_text], test_labels, verbose=1)

# Extract and print accuracy on the training set from the history
train_accuracy = history.history['acc'][-1]
print(f"Accuracy on training set (End of Training): {train_accuracy}")

from sklearn.metrics import classification_report

# Predictions on the test set
test_pred = model_BiLstm.predict([np.array(test_title), np.array(test_text)])

# Convert predictions to categorical format
test_pred_categorical = np.argmax(test_pred, axis=1)

# Calculate and print classification report
report = classification_report(np.argmax(test_labels, axis=1), test_pred_categorical)
print("Classification Report on Test Set of BiLSTM:\n", report)


# BiLSTM + CNN Titles + Contents
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Conv1D, MaxPooling1D, Dense, Dropout, Concatenate, Flatten

filter_sizes = [3, 4, 5]
num_filters = 100
drop = 0.5

embedding_dim = 50  # Adjust the dimensionality as needed
max_words = w2vModel.vectors.shape[0]

def convolutional_block(input_layer, max_len, filter_sizes):
    conv_blocks = []
    for filter_size in filter_sizes:
        conv = Conv1D(filters=num_filters,
                      kernel_size=filter_size,
                      padding='valid',
                      activation='relu')(input_layer)
        maxpool = MaxPooling1D(pool_size=(max_len - filter_size + 1))(conv)
        conv_blocks.append(maxpool)

    return Concatenate(axis=-1)(conv_blocks)

def build_classifier_model_bilstm_cnn_title_content():
    # Titles
    inputs_title = Input(shape=(max_len_title,), dtype='int32', name='input_title')
    embedding_title = Embedding(max_words, embedding_dim,
                                weights=[w2vModel.vectors[:max_words, :embedding_dim]],
                                input_length=max_len_title,
                                trainable=True)(inputs_title)
    lstm_title = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(embedding_title)
    title_conv_block = convolutional_block(lstm_title, max_len_title, filter_sizes)

    # Text
    inputs_text = Input(shape=(max_len_text,), dtype='int32', name='input_text')
    embedding_text = Embedding(max_words, embedding_dim,
                               weights=[w2vModel.vectors[:max_words, :embedding_dim]],
                               input_length=max_len_text,
                               trainable=True)(inputs_text)
    lstm_text = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(embedding_text)
    text_conv_block = convolutional_block(lstm_text, max_len_text, filter_sizes)

    # Fully connected layers for title and text with dropout
    fc_title = Dense(128, activation='relu')(title_conv_block)
    fc_title = Dropout(drop)(fc_title)

    fc_text = Dense(128, activation='relu')(text_conv_block)
    fc_text = Dropout(drop)(fc_text)

    # Concatenate the outputs
    concatenated_output = Concatenate(axis=-1)([fc_title, fc_text])
    flattened_output = Flatten()(concatenated_output)
    dropout_output = Dropout(drop)(flattened_output)

    preds = Dense(3, activation='softmax', name='classifier')(dropout_output)

    # This creates a model that includes inputs and outputs
    model_BiLstm_CNN = tf.keras.Model(inputs=[inputs_title, inputs_text], outputs=preds)

    model_BiLstm_CNN.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model_BiLstm_CNN

model_BiLstm_CNN = build_classifier_model_bilstm_cnn_title_content()
model_BiLstm_CNN.summary()

model_BiLstm_CNN.save_weights('BiLSTM+CNN.h5')

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow as tf

# Function for learning rate scheduling
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


    model_BiLstm_CNN = build_classifier_model_bilstm_cnn_title_content()

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_schedule = LearningRateScheduler(lr_scheduler)

# Train the model with callbacks
history = model_BiLstm_CNN.fit(
    [train_title, train_text],
    train_labels,
    batch_size=4,
    epochs=100,
    validation_split=0.10,
    verbose=1,
    callbacks=[early_stopping, lr_schedule]
)

# Evaluate the model on the test set
score = model_BiLstm_CNN.evaluate([test_title, test_text], test_labels, batch_size=4, verbose=1)

# Extract and print accuracy on the training set from the history
train_accuracy = history.history['acc'][-1]
print(f"Accuracy on training set (End of Training): {train_accuracy}")

# Predictions on the test set
test_pred = model_BiLstm_CNN.predict([np.array(test_title), np.array(test_text)])

# Convert predictions to categorical format
test_pred_categorical = np.argmax(test_pred, axis=1)

# Calculate and print classification report
report = classification_report(np.argmax(test_labels, axis=1), test_pred_categorical)
print("Classification Report on Test Set of BiLSTM + CNN:\n", report)


# LSTM + CNN Ttles + Contents
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Conv2D, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, GlobalMaxPooling2D
from tensorflow.keras.models import Model

filter_sizes = [3, 4, 5]
num_filters = 150
drop = 0.7

def build_classifier_model_lstm_cnn_title_content():
    # Titles
    inputs_title = Input(shape=(max_len_title,), dtype='int32')
    embedding_title = Embedding(w2vModel.vectors.shape[0], w2vModel.vectors.shape[1], weights=[w2vModel.vectors],
                                input_length=max_len_title, trainable=False)(inputs_title)
    lstm_title = LSTM(256, activation='tanh', return_sequences=True)(embedding_title)
    reshape_title = Reshape((max_len_title, 256, 1))(lstm_title)

    conv_title_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 256), padding='valid', kernel_initializer='normal', activation='relu')(reshape_title)
    conv_title_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 256), padding='valid', kernel_initializer='normal', activation='relu')(reshape_title)
    conv_title_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], 256), padding='valid', kernel_initializer='normal', activation='relu')(reshape_title)

    maxpool_title_0 = GlobalMaxPooling2D()(conv_title_0)
    maxpool_title_1 = GlobalMaxPooling2D()(conv_title_1)
    maxpool_title_2 = GlobalMaxPooling2D()(conv_title_2)

    concatenated_title_tensor = Concatenate(axis=1)([maxpool_title_0, maxpool_title_1, maxpool_title_2])
    dropout_title = Dropout(drop)(concatenated_title_tensor)

    # Text
    inputs_text = Input(shape=(max_len_text,), dtype='int32')
    embedding_text = Embedding(w2vModel.vectors.shape[0], w2vModel.vectors.shape[1], weights=[w2vModel.vectors],
                               input_length=max_len_text, trainable=False)(inputs_text)
    lstm_text = LSTM(256, activation='tanh', return_sequences=True)(embedding_text)
    reshape_text = Reshape((max_len_text, 256, 1))(lstm_text)

    conv_text_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 256), padding='valid', kernel_initializer='normal', activation='relu')(reshape_text)
    conv_text_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 256), padding='valid', kernel_initializer='normal', activation='relu')(reshape_text)
    conv_text_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], 256), padding='valid', kernel_initializer='normal', activation='relu')(reshape_text)

    maxpool_text_0 = GlobalMaxPooling2D()(conv_text_0)
    maxpool_text_1 = GlobalMaxPooling2D()(conv_text_1)
    maxpool_text_2 = GlobalMaxPooling2D()(conv_text_2)

    concatenated_text_tensor = Concatenate(axis=1)([maxpool_text_0, maxpool_text_1, maxpool_text_2])
    dropout_text = Dropout(drop)(concatenated_text_tensor)

    # Average outputs
    average = Concatenate(axis=1)([dropout_title, dropout_text])

    preds = Dense(3, activation='softmax', name='classifier')(average)

    # Tạo mô hình với inputs và outputs
    model_lstm_cnn = Model(inputs=[inputs_title, inputs_text], outputs=preds)

    model_lstm_cnn.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model_lstm_cnn

# Sử dụng hàm để xây dựng mô hình
model_lstm_cnn = build_classifier_model_lstm_cnn_title_content()

# In thông tin mô hình
model_lstm_cnn.summary()

model_lstm_cnn.save_weights('LSTM+CNN.h5')

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow as tf

# Function for learning rate scheduling
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

model_lstm_cnn = build_classifier_model_lstm_cnn_title_content()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = LearningRateScheduler(lr_scheduler)

# Train the model with callbacks
history = model_lstm_cnn.fit(
    [train_title, train_text],
    train_labels,
    batch_size=4,
    epochs=100,
    validation_split=0.10,
    verbose=1,
    callbacks=[early_stopping, lr_schedule]
)

# Evaluate the model on the test set
score = model_lstm_cnn.evaluate([test_title, test_text], test_labels, batch_size=4, verbose=1)

# Extract and print accuracy on the training set from the history
train_accuracy = history.history['acc'][-1]
print(f"Accuracy on training set (End of Training): {train_accuracy}")

# Predictions on the test set
test_pred = model_lstm_cnn.predict([np.array(test_title), np.array(test_text)])

# Convert predictions to categorical format
test_pred_categorical = np.argmax(test_pred, axis=1)

# Calculate and print classification report
report = classification_report(np.argmax(test_labels, axis=1), test_pred_categorical)
print("Classification Report on Test Set:\n", report)

# CNN Contents+Titles
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Conv2D, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam

filter_sizes = [3, 4, 5]
num_filters = 150
drop = 0.7

def build_classifier_model_cnn_title_content():
    # Process Title
    input_title = Input(shape=(max_len_title,), dtype='int32')
    embedding_title = Embedding(w2vModel.vectors.shape[0], w2vModel.vectors.shape[1], weights=[w2vModel.vectors], input_length=max_len_title, trainable=False)(input_title)
    reshape_title = Reshape((max_len_title, 300, 1))(embedding_title)

    conv_title_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 300), padding='valid', kernel_initializer='normal', activation='relu')(reshape_title)
    conv_title_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 300), padding='valid', kernel_initializer='normal', activation='relu')(reshape_title)
    conv_title_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], 300), padding='valid', kernel_initializer='normal', activation='relu')(reshape_title)

    maxpool_title_0 = MaxPool2D(pool_size=(max_len_title - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_title_0)
    maxpool_title_1 = MaxPool2D(pool_size=(max_len_title - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_title_1)
    maxpool_title_2 = MaxPool2D(pool_size=(max_len_title - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_title_2)

    concatenated_title_tensor = Concatenate(axis=1)([maxpool_title_0, maxpool_title_1, maxpool_title_2])
    flatten_title = Flatten()(concatenated_title_tensor)
    dropout_title = Dropout(drop)(flatten_title)
    batch_norm_title = BatchNormalization()(dropout_title)

    # Process Text
    inputs_text = Input(shape=(max_len_text,), dtype='int32')
    embedding_text = Embedding(w2vModel.vectors.shape[0], w2vModel.vectors.shape[1], weights=[w2vModel.vectors], input_length=max_len_text, trainable=False)(inputs_text)
    reshape_text = Reshape((max_len_text, 300, 1))(embedding_text)

    conv_text_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 300), padding='valid', kernel_initializer='normal', activation='relu')(reshape_text)
    conv_text_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 300), padding='valid', kernel_initializer='normal', activation='relu')(reshape_text)
    conv_text_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], 300), padding='valid', kernel_initializer='normal', activation='relu')(reshape_text)

    maxpool_text_0 = MaxPool2D(pool_size=(max_len_text - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_text_0)
    maxpool_text_1 = MaxPool2D(pool_size=(max_len_text - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_text_1)
    maxpool_text_2 = MaxPool2D(pool_size=(max_len_text - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_text_2)

    concatenated_text_tensor = Concatenate(axis=1)([maxpool_text_0, maxpool_text_1, maxpool_text_2])
    flatten_text = Flatten()(concatenated_text_tensor)
    dropout_text = Dropout(drop)(flatten_text)
    batch_norm_text = BatchNormalization()(dropout_text)

    # Dropout layers for regularization
    dropout_title = Dropout(drop)(flatten_title)
    dropout_text = Dropout(drop)(flatten_text)

    # Concatenation outputs
    average = Concatenate(axis=1)([dropout_title, dropout_text])
    out_put = Dense(units=3, activation='softmax', name='classifier')(average)

    model_CNN = tf.keras.Model(inputs=[input_title, inputs_text], outputs=out_put)

    # Use a lower learning rate
    optimizer = Adam(lr=0.0001)

    model_CNN.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    return model_CNN

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model_CNN = build_classifier_model_cnn_title_content()
model_CNN.summary()

model_CNN.save_weights('CNN.h5')

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow as tf

# Function for learning rate scheduling
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

model_CNN = build_classifier_model_cnn_title_content()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = LearningRateScheduler(lr_scheduler)

# Train the model with callbacks
history = model_CNN.fit(
    [train_title, train_text],
    train_labels,
    batch_size=4,
    epochs=100,
    validation_split=0.10,
    verbose=1,
    callbacks=[early_stopping, lr_schedule]
)

# Evaluate the model on the test set
score = model_CNN.evaluate([test_title, test_text], test_labels, batch_size=4, verbose=1)

# Extract and print accuracy on the training set from the history
train_accuracy = history.history['acc'][-1]
print(f"Accuracy on training set (End of Training): {train_accuracy}")

from sklearn.metrics import classification_report

# Predictions on the test set
test_pred = model_CNN.predict([np.array(test_title), np.array(test_text)])

# Convert predictions to categorical format
test_pred_categorical = np.argmax(test_pred, axis=1)

# Calculate and print classification report
report = classification_report(np.argmax(test_labels, axis=1), test_pred_categorical)
print("Classification Report on Test Set:\n", report)

# LSTM Titles + Contents
def build_classifier_model_lstm_title_content():
    title = Input(shape=(30,), name='Review_Title')
    embedding_title = Embedding(w2vModel.vectors.shape[0], w2vModel.vectors.shape[1], weights=[w2vModel.vectors], input_length=max_len_title, trainable=False)(title)
    lstm_title = LSTM(128, activation='tanh')(embedding_title)

    text = Input(shape=(150,), name='Review_Text')
    embedding_text = Embedding(w2vModel.vectors.shape[0], w2vModel.vectors.shape[1], weights=[w2vModel.vectors], input_length=max_len_text, trainable=False)(text)
    lstm_text = LSTM(128, activation='tanh')(embedding_text)

    average = Concatenate(axis=1)([lstm_title, lstm_text])
    output = Dense(units=5, activation='softmax', name='classifier')(average)

    model_LSTM = tf.keras.Model(inputs=[title, text], outputs=output)
    model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_LSTM

model_LSTM = build_classifier_model_lstm_title_content()
model_LSTM.summary()

model_LSTM.save_weights('LSTM.h5')

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow as tf

# Function for learning rate scheduling
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

model_LSTM = build_classifier_model_lstm_title_content()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = LearningRateScheduler(lr_scheduler)

# Train the model with callbacks
history = model_LSTM.fit(
    [train_title, train_text],
    train_labels,
    batch_size=4,
    epochs=100,
    validation_split=0.10,
    verbose=1,
    callbacks=[early_stopping, lr_schedule]
)

# Evaluate the model on the test set
score = model_LSTM.evaluate([test_title, test_text], test_labels, batch_size=4, verbose=1)

# Extract and print accuracy on the training set from the history
train_accuracy = history.history['accuracy'][-1]
test_accuracy = score[1]
print(f"Accuracy on training set (End of Training): {train_accuracy}")
print(f"Accuracy on test set: {test_accuracy}")

# Predictions on the test set
test_pred = model_LSTM.predict([np.array(test_title), np.array(test_text)])

# Convert predictions to categorical format
test_pred_categorical = np.argmax(test_pred, axis=1)

# Calculate and print classification report
report = classification_report(np.argmax(test_labels, axis=1), test_pred_categorical)
print("Classification Report on Test Set:\n", report)

