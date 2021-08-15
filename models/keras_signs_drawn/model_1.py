import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from sklearn.metrics import accuracy_score

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from google.colab import drive


now_drive = '/content/drive6'
drive.mount(now_drive)

directory = now_drive + '/My Drive/Colab Notebooks/ML/All_train/'
df = pd.read_csv(directory + 'train.csv')
file_paths = df['file_name'].values
labels = df['label'].values

def shuffle_in_unison(a, b):
     n_elem = a.shape[0]
     indeces = np.random.choice(n_elem, size=n_elem, replace=False)
     return a[indeces], b[indeces]
  
  
def read_image(img_path, label):
    image = tf.io.read_file(directory + img_path)
    image = tf.image.decode_png(image, channels=1, dtype=tf.uint8)
    image = tf.image.resize(image, [40, 30])
    return image, label
  

file_paths, labels = shuffle_in_unison(file_paths, labels)
ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))

image_w = 30
image_h = 40
batch_size = 32

ds_train = ds_train.map(read_image).batch(batch_size)

# setting up model

model = keras.Sequential([
    keras.layers.Input((image_h, image_w, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='sigmoid'),
    keras.layers.Dense(32, activation='sigmoid'),
    keras.layers.Dense(3, activation='sigmoid'),
])

# compiling

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[
          keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    ],
    metrics='accuracy'
)

# fittings

model.fit(ds_train, epochs=15, verbose=1)

# testing

directory_test = now_drive + '/My Drive/Colab Notebooks/ML/All_test/'
df_test = pd.read_csv(directory_test + 'test.csv')
file_paths_test = df_test['file_name'].values
labels_test = df_test['label'].values

ds_test = tf.data.Dataset.from_tensor_slices((file_paths_test, labels_test))
ds_test = ds_test.map(read_image).batch(batch_size)

prediction_np = np.argmax(model.predict(ds_test), axis=1)

accuracy_score(labels_test, prediction_np)

# saving

model.save(now_drive + '/My Drive/Colab Notebooks/ML/Models/model.h5')

drive.flush_and_unmount()
