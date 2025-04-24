from typing import Text
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras.datasets import imdb

sample = ['The cat sat on the mat.', 'The dog ate my homework!']

vectorize_layer= TextVectorization(max_tokens=10, output_mode='int')
vectorize_layer.adapt(sample)

#print(vectorize_layer.get_vocabulary(include_special_tokens=True))
vectorized_text = vectorize_layer(sample)
#print(vectorized_text)

embedding_layer = Embedding(20,2)
text_embedding = embedding_layer(vectorized_text)
#print(text_embedding)

max_features = 1000
max_words =1000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)
train_data = tf.keras.utils.pad_sequences(train_data, maxlen=max_words)
test_data = tf.keras.utils.pad_sequences(test_data, maxlen=max_words)

#(train_data, train_labels), (test_data, test_labels) = imdb.load_data()
#print(len(train_data[3]))

#model = tf.keras.models.Sequential(
#    [tf.keras.layers.Embedding(max_features, 64),
#    tf.keras.layers.Flatten(),
#    tf.keras.layers.Dense(16, activation='relu'),
#    tf.keras.layers.Dense(1, activation='sigmoid')]
#    )

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

#history = model.fit(train_data, train_labels, epochs=20, validation_split=0.2)

model = tf.keras.models.Sequential(
    [tf.keras.layers.Embedding(max_features, 64),
    tf.keras.layers.SimpleRNN(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')]
    )

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2, batch_size=256)

model.evaluate(test_data, test_labels)
print(summarize_diagnostics(history, 'accuracy'))