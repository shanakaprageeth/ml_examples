import os
from six.moves import urllib
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt

# Examine software versions
print(__import__('sys').version)
print(tf.__version__)
print(tf.keras.__version__)

### For downloading data ###

# Storage directory
DATA_DIR = os.path.join(tempfile.gettempdir(), 'census_data')

# Download options.
DATA_URL = 'https://storage.googleapis.com/cloud-samples-data/ai-platform' \
           '/census/data'
TRAINING_FILE = 'adult.data.csv'
EVAL_FILE = 'adult.test.csv'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

### For interpreting data ###

# These are the features in the dataset.
# Dataset information: https://archive.ics.uci.edu/ml/datasets/census+income
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CATEGORICAL_TYPES = {
  'workclass': pd.api.types.CategoricalDtype(categories=[
    'Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc',
    'Self-emp-not-inc', 'State-gov', 'Without-pay'
  ]),
  'marital_status': pd.api.types.CategoricalDtype(categories=[
    'Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
    'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'
  ]),
  'occupation': pd.api.types.CategoricalDtype([
    'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
    'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct',
    'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv',
    'Sales', 'Tech-support', 'Transport-moving'
  ]),
  'relationship': pd.api.types.CategoricalDtype(categories=[
    'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried',
    'Wife'
  ]),
  'race': pd.api.types.CategoricalDtype(categories=[
    'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'
  ]),
  'native_country': pd.api.types.CategoricalDtype(categories=[
    'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic',
    'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece',
    'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary',
    'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico',
    'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland',
    'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand',
    'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia'
  ]),
  'income_bracket': pd.api.types.CategoricalDtype(categories=[
    '<=50K', '>50K'
  ])
}

# This is the label (target) we want to predict.
_LABEL_COLUMN = 'income_bracket'

### Hyperparameters for training ###

# This the training batch size
BATCH_SIZE = 128

# This is the number of epochs (passes over the full training data)
NUM_EPOCHS = 20

# Define learning rate.
LEARNING_RATE = .01

def _download_and_clean_file(filename, url):
  """Downloads data from url, and makes changes to match the CSV format.

  The CSVs may use spaces after the comma delimters (non-standard) or include
  rows which do not represent well-formed examples. This function strips out
  some of these problems.

  Args:
    filename: filename to save url to
    url: URL of resource to download
  """
  temp_file, _ = urllib.request.urlretrieve(url)
  with tf.gfile.Open(temp_file, 'r') as temp_file_object:
    with tf.gfile.Open(filename, 'w') as file_object:
      for line in temp_file_object:
        line = line.strip()
        line = line.replace(', ', ',')
        if not line or ',' not in line:
          continue
        if line[-1] == '.':
          line = line[:-1]
        line += '\n'
        file_object.write(line)
  tf.gfile.Remove(temp_file)


def download(data_dir):
  """Downloads census data if it is not already present.

  Args:
    data_dir: directory where we will access/save the census data
  """
  tf.gfile.MakeDirs(data_dir)

  training_file_path = os.path.join(data_dir, TRAINING_FILE)
  if not tf.gfile.Exists(training_file_path):
    _download_and_clean_file(training_file_path, TRAINING_URL)

  eval_file_path = os.path.join(data_dir, EVAL_FILE)
  if not tf.gfile.Exists(eval_file_path):
    _download_and_clean_file(eval_file_path, EVAL_URL)

  return training_file_path, eval_file_path


training_file_path, eval_file_path = download(DATA_DIR)

# This census data uses the value '?' for fields (column) that are missing data.
# We use na_values to find ? and set it to NaN values.
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

train_df = pd.read_csv(training_file_path, names=_CSV_COLUMNS, na_values='?')
eval_df = pd.read_csv(eval_file_path, names=_CSV_COLUMNS, na_values='?')

UNUSED_COLUMNS = ['fnlwgt', 'education', 'gender']


def preprocess(dataframe):
  """Converts categorical features to numeric. Removes unused columns.

  Args:
    dataframe: Pandas dataframe with raw data

  Returns:
    Dataframe with preprocessed data
  """
  dataframe = dataframe.drop(columns=UNUSED_COLUMNS)

  # Convert integer valued (numeric) columns to floating point
  numeric_columns = dataframe.select_dtypes(['int64']).columns
  dataframe[numeric_columns] = dataframe[numeric_columns].astype('float32')

  # Convert categorical columns to numeric
  cat_columns = dataframe.select_dtypes(['object']).columns
  dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.astype(
    _CATEGORICAL_TYPES[x.name]))
  dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.cat.codes)
  return dataframe

prepped_train_df = preprocess(train_df)
prepped_eval_df = preprocess(eval_df)

# Split train and test data with labels.
# The pop() method will extract (copy) and remove the label column from the dataframe
train_x, train_y = prepped_train_df, prepped_train_df.pop(_LABEL_COLUMN)
eval_x, eval_y = prepped_eval_df, prepped_eval_df.pop(_LABEL_COLUMN)

# Reshape label columns for use with tf.data.Dataset
train_y = np.asarray(train_y).astype('float32').reshape((-1, 1))
eval_y = np.asarray(eval_y).astype('float32').reshape((-1, 1))

def standardize(dataframe):
  """Scales numerical columns using their means and standard deviation to get
  z-scores: the mean of each numerical column becomes 0, and the standard
  deviation becomes 1. This can help the model converge during training.

  Args:
    dataframe: Pandas dataframe

  Returns:
    Input dataframe with the numerical columns scaled to z-scores
  """
  dtypes = list(zip(dataframe.dtypes.index, map(str, dataframe.dtypes)))
  # Normalize numeric columns.
  for column, dtype in dtypes:
      if dtype == 'float32':
          dataframe[column] -= dataframe[column].mean()
          dataframe[column] /= dataframe[column].std()
  return dataframe


# Join train_x and eval_x to normalize on overall means and standard
# deviations. Then separate them again.
all_x = pd.concat([train_x, eval_x], keys=['train', 'eval'])
all_x = standardize(all_x)
train_x, eval_x = all_x.xs('train'), all_x.xs('eval')


def input_fn(features, labels, shuffle, num_epochs, batch_size):
  """Generates an input function to be used for model training.

  Args:
    features: numpy array of features used for training or inference
    labels: numpy array of labels for each example
    shuffle: boolean for whether to shuffle the data or not (set True for
      training, False for evaluation)
    num_epochs: number of epochs to provide the data for
    batch_size: batch size for training

  Returns:
    A tf.data.Dataset that can provide data to the Keras model for training or
      evaluation
  """
  if labels is None:
    inputs = features
  else:
    inputs = (features, labels)
  dataset = tf.data.Dataset.from_tensor_slices(inputs)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=len(features))

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset

# Pass a numpy array by using DataFrame.values
training_dataset = input_fn(features=train_x.values,
                    labels=train_y,
                    shuffle=True,
                    num_epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE)

num_eval_examples = eval_x.shape[0]

# Pass a numpy array by using DataFrame.values
validation_dataset = input_fn(features=eval_x.values,
                    labels=eval_y,
                    shuffle=False,
                    num_epochs=NUM_EPOCHS,
                    batch_size=num_eval_examples)

def create_keras_model(input_dim, learning_rate):
  """Creates Keras Model for Binary Classification.

  Args:
    input_dim: How many features the input has
    learning_rate: Learning rate for training

  Returns:
    The compiled Keras model (still needs to be trained)
  """
  Dense = tf.keras.layers.Dense
  model = tf.keras.Sequential(
    [
        Dense(100, activation=tf.nn.relu, kernel_initializer='uniform',
                input_shape=(input_dim,)),
        Dense(75, activation=tf.nn.relu),
        Dense(50, activation=tf.nn.relu),
        Dense(25, activation=tf.nn.relu),
        Dense(1, activation=tf.nn.sigmoid)
    ])

  # Custom Optimizer:
  # https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
  optimizer = tf.keras.optimizers.RMSprop(
      lr=learning_rate)

  # Compile Keras model
  model.compile(
      loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

num_train_examples, input_dim = train_x.shape
print('Number of features: {}'.format(input_dim))
print('Number of examples: {}'.format(num_train_examples))

keras_model = create_keras_model(
    input_dim=input_dim,
    learning_rate=LEARNING_RATE)

keras_model.summary()

# Setup Learning Rate decay.
lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: LEARNING_RATE + 0.02 * (0.5 ** (1 + epoch)),
    verbose=True)

# Setup TensorBoard callback.
JOB_DIR = os.getenv('JOB_DIR')
tensorboard_cb = tf.keras.callbacks.TensorBoard(
      os.path.join(JOB_DIR, 'keras_tensorboard'),
      histogram_freq=1)

history = keras_model.fit(training_dataset,
                          epochs=NUM_EPOCHS,
                          steps_per_epoch=int(num_train_examples/BATCH_SIZE),
                          validation_data=validation_dataset,
                          validation_steps=1,
                          callbacks=[lr_decay_cb, tensorboard_cb],
                          verbose=1)


# Visualize History for Loss.
plt.title('Keras model loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

# Visualize History for Accuracy.
plt.title('Keras model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc='lower right')
plt.show()

# Export the model to a local SavedModel directory
export_path = tf.contrib.saved_model.save_keras_model(keras_model, 'keras_export')
print("Model exported to: ", export_path)

#JOB_DIR = os.getenv('JOB_DIR')

# Export the model to a SavedModel directory in Cloud Storage
#export_path = tf.contrib.saved_model.save_keras_model(keras_model, JOB_DIR + '/keras_export')
#print("Model exported to: ", export_path)
