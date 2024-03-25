import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm


def download_mnist_sign_language():
  kaggle_command = f'kaggle datasets download -d datamunge/sign-language-mnist'


# %%
def load_csv_dataset(csv_path):
  df = pd.read_csv(csv_path)
  return df

def load_mnist_sign_dataset():
  dataset_directory = 'data/archive/'
  test_set_path = f'{dataset_directory}/sign_mnist_test/sign_mnist_test.csv'
  train_set_path = f'{dataset_directory}/sign_mnist_train/sign_mnist_train.csv'
  df_test = pd.read_csv(test_set_path)
  df_train = pd.read_csv(train_set_path)
  # df_train.info()
  # df_test.info()
  return df_test, df_train

def shape(df):
  Y = df['label']
  lb = LabelBinarizer()
  Y = lb.fit_transform(Y)
  # the dataset only contains 24 labels
  # but there are 26 letters in the alphabet
  # the ones that are missing is J and Z
  # we will add a row of zeros for J and Z
  # to make the labels 26
  # they should be inserted in the corresponing row
  # J is 9 and Z is 25
  Y = np.insert(Y, 9, 0, axis=1)
  Y = np.insert(Y, 25, 0, axis=1)


  X = df.drop(['label'],axis=1)
  X = X.values.reshape(-1,28,28,1)
  # one hot encode

  # print the shape of Y
  print(f'Y shape: {Y.shape}')

  # flatten the images
  X = X.reshape(X.shape[0], -1)
  return X, Y

def standardize_data(X_t, X_v, X_e):
  # center by subtracting the mean
  # divide by the standard deviation
  # N X D 
  # N X W X H X C
  # copy the data
  max_val = 255 # max val for pixel data is 255
  X_t_bar = X_t.copy() / max_val
  X_v_bar = X_v.copy() / max_val
  X_e_bar = X_e.copy() / max_val

  X_t_bar_mean = X_t_bar.mean(axis=0)
  X_t_bar_std = X_t_bar.std(axis=0)

  X_t_bar = (X_t_bar - X_t_bar_mean) / X_t_bar_std
  X_v_bar = (X_v_bar - X_t_bar_mean) / X_t_bar_std
  X_e_bar = (X_e_bar - X_t_bar_mean) / X_t_bar_std
  return X_t_bar, X_v_bar, X_e_bar



def display_image(X, Y, index):
  plt.imshow(X[index].reshape(28,28), cmap='gray')
  alphanumeric = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  plt.title(f'{alphanumeric[np.argmax(Y[index])]}')
  plt.show()

def display_images(X, Y, indices, title=None):
  # do a multiplot
  # that displays the the first 9 images
  # with their labels
  fig, ax = plt.subplots(3, 3, figsize=(10,10))
  # add title
  if title:
    fig.suptitle(title)
  alphanumeric = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  for i in range(3):
    for j in range(3):
      ax[i,j].imshow(X[indices[i*3+j]].reshape(28,28), cmap='gray')
      ax[i,j].set_title(f'{indices[i*3+j]}: {alphanumeric[np.argmax(Y[indices[i*3+j]])]}')
  plt.show()

def shuffle_and_split(df, split_ratio=0.8):
  # shuffle the data
  df = df.sample(frac=1).reset_index(drop=True)
  # split the data
  split_index = int(df.shape[0] * split_ratio)
  train = df.iloc[:split_index]
  test = df.iloc[split_index:]
  return train, test

  

test, train = load_mnist_sign_dataset()

train, val = shuffle_and_split(train, 0.8)

X_e, Y_e = shape(test)
X_t, Y_t = shape(train)
X_v, Y_v = shape(val)


# display first 9 images of the test set
# display_images(X_e, Y_e, [0,1,2,3,4,5,6,7,8], 'Test Set')
# display_images(X_t, Y_t, [0,1,2,3,4,5,6,7,8], 'Train Set')
# display_images(X_v, Y_v, [0,1,2,3,4,5,6,7,8], 'Validation Set')
X_t, X_v, X_e = standardize_data(X_t, X_v, X_e)

# set data type to float32
# the values can only be 255 different values
# that about 8 bits
# should not need more that per pixel because its a grayscale image
# X_t = X_t.astype(np.float32)
# X_v = X_v.astype(np.float32)
# X_e = X_e.astype(np.float32)

#display first 5 images
# display_images(X_e, Y_e, [0,1,2,3,4,5,6,7,8], 'Test Set Standardized')
# display_images(X_t, Y_t, [0,1,2,3,4,5,6,7,8], 'Train Set Standardized')

# print the dimensions
print(f'Test Set: X:{X_e.shape} Y:{Y_e.shape}')
print(f'Train Set: X:{X_t.shape} Y:{Y_t.shape}')

import tensorflow as tf


# MLP construction

def build_mlp(input_shape, output_shape, width, depth, activation='relu'):

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Input(shape=(input_shape,)))
  for i in range(depth):
    model.add(tf.keras.layers.Dense(width, activation=activation))
  model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
  return model

def train_mlp(model, X, Y, X_val, Y_val, epochs, batch_size, lr):
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  history = model.fit(X, Y, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size)
  return history

def plot_history(histories, title='Model Loss and Accuracy', plot_file_path=None):
    fig, ax1 = plt.subplots()

    # Set up the first axis for loss
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    smoothing = 0.1


    # Plot each loss history
    for label, data in histories['loss']:
        ax1.plot(range(1, len(data) + 1), data, label=f'{label} Loss', color='tab:red' if 'train' in label else 'lightcoral')

    # Set up the second axis for accuracy
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Accuracy', color='tab:blue') 
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Plot each accuracy history
    for label, data in histories['accuracy']:
        # multiply the data by 100 to convert to percentage
        data = [x * 100 for x in data]
        ax2.plot(range(1, len(data) + 1), data, label=f'{label} Accuracy', color='tab:blue' if 'train' in label else 'lightblue')

    # Add legends and title
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax2.set_ylim([0, 100])
    plt.title(title)

    # Save the plot to a file if a file path is provided, else display the plot
    if plot_file_path:
        plt.savefig(plot_file_path)
        plt.close(fig)  # Close the figure to avoid displaying it in the notebook/output
    else:
        plt.show()

def experiment1_benchmark(model_params, folder_name):

  input_shape = X_t.shape[1]
  output_shape = Y_t.shape[1]

  # model_params = [
  #   (512, 3),
  #   (32, 0),
  #   (32, 1),
  #   (64, 1),
  #   (128, 1),
  #   (256, 1),
  #   (32, 2),
  #   (64, 2),
  #   (128, 2),
  #   (256, 2)
  # ]
  # model_params = []
  # for i in range(0, 5):
  #   for j in range(4, 10):
  #     model_params.append((2**j, i))


  test_results = {}

  for model_param in model_params:
    width, depth = model_param
    model = build_mlp(input_shape, output_shape, width, depth)
    history = train_mlp(model, X_t, Y_t, X_v, Y_v, epochs=20, batch_size=32, lr=0.01)

    histories = {
      'loss': [
        ('train', history.history['loss']),
        ('val', history.history['val_loss'])
      ],
      'accuracy': [
        ('train', history.history['accuracy']),
        ('val', history.history['val_accuracy'])
      ]
    }

    plot_history(histories, title=f'Width: {width}, Depth: {depth}', plot_file_path=f'figures/{folder_name}/width_{depth}_{width}.png')
    loss, accuracy = model.evaluate(X_e, Y_e)
    print(f'Width: {width}, Depth: {depth}, Loss: {loss}, Accuracy: {accuracy}')
    test_results[(width, depth)] = (loss, accuracy)

  # plot the results where accuracy is on the y-axis and width is on the x-axis
  # there should be a line for each depth
  # for the one with 0 depth, just continue the line across the x-axis for the single point
  
  # clear the  sub plots
  

  fig, ax = plt.subplots()

  # pull out the all the depths
  depths = set([depth for width, depth in model_params])

  for depth in depths:
    x = [width for width, depth_ in model_params if depth_ == depth]
    y = [test_results[(width, depth)][1] for width in x]
    ax.plot(x, y, label=f'Depth: {depth}')
  ax.legend()
  # label the x and y axis
  ax.set_xlabel('Width')
  ax.set_ylabel('Accuracy')
  # save plot
  plt.savefig(f'figures/{folder_name}/accuracy_vs_width.png')
  plt.close(fig)
  
# model_params = []
# for i in range(0, 5):
#   for j in range(4, 10):
#     model_params.append((2**j, i))

# experiment1_benchmark(model_params, 'exp1_bench_1_')

# model_params = []
# for i in range(1, 5):
#   for j in range(0, 200, 20):
#     model_params.append((j, i))

# # reverse the order of the model params
# model_params = model_params[::-1]

# experiment1_benchmark(model_params, 'exp1_bench_6') 

# lr bench 2: 0.0001
# lr bench 3: 0.00001 , standardization on only the training set
# lr bench 4: 0.00001 , no standardization
# lr bench 5: 0.001 , standardization on the training and validation set and test set
# lr bench 6: 0.001 , standardization pixel wise, not on the entire dataset
# lr bench 7: 0.0001 , standardization pixel wise, not on the entire dataset


#================================================================================================
# MLP implementation
#================================================================================================

# Requirements:
# softmax
# relu
# cross entropy loss
# softmax derivative
# relu derivative
# cross entropy loss derivative
# Dense layer
# Relu Layer
# Softmax Layer
# Model constructor - architecture
# Model compiler - training params
# Model fit training loop
# model evaluation

# vectorised calculations

# UTIL FUNCTIONS

def softmax(X):
  # vectorised softmax function
  # softmax function
  # X: N X C
  # N is the number of samples
  # C is the number of classes
  # we need to make sure this is numerically stable
  # subtract the maximum value from each row
  # this will make the largest value 0
  # this will prevent overflow
  exp = np.exp(X - np.max(X, axis=1, keepdims=True))
  return exp / exp.sum(axis=1, keepdims=True)

def relu(X):
  # vectorised relu function
  # relu function
  # X: N X D
  # N is the number of samples
  # D is the number of dimensions
  return np.maximum(X, 0)

def cross_entropy_loss(Y, Y_hat):
  # vectorised cross entropy loss function
  # cross entropy loss
  # Y: N X C
  # Y_hat: N X C
  # N is the number of samples
  # C is the number of classes
  # we need to make sure this is numerically stable
  # add a small number to prevent log(0)
  # raise an error if any of the values are zero
  if (Y_hat == 0).any():
    raise ValueError('Y_hat contains zero values which will cause log(0) to be undefined')

  return -np.sum(Y * np.log(Y_hat)) / Y.shape[0]

def d_relu(X):
  # vectorised relu derivative function
  # relu derivative
  # X: N X D
  # N is the number of samples
  # D is the number of dimensions
  return np.where(X > 0, 1, 0)

# LAYERS

class DenseLayer():

  def __init__(self, input_shape, output_shape, weights=None, biases=None):
    # initialise the weights and biases
    # input_shape: D
    # output_shape: M
    # D is the number of input dimensions
    # M is the number of output dimensions

    # this is the common way to initialise the weights for ReLU
    self.D = input_shape
    self.M = output_shape

    # M IS THE NUMBER OF UNITS IN THE LAYER
    # AKA THE OUTPUT SHAPE
    # NOT TO BE CONFUSED WITH NUMBER OF UNITS IN THE OUTPUT LAYER

    self.W = np.random.randn(input_shape, output_shape) * np.sqrt(2/input_shape)
    self.b = np.zeros(output_shape) + 0.01

    if weights is not None:
      # check the shape is correct
      if weights.shape != (input_shape, output_shape):
        raise ValueError('Weights shape is incorrect')
      self.W = weights
    if biases is not None:
      # check the shape is correct
      if biases.shape != (output_shape,):
        raise ValueError('Biases shape is incorrect')
      self.b = biases

  def forward(self, X):

    # forward pass
    # X: N X D
    # N is the number of samples
    # D is the number of dimensions
    # return the result of the forward pass
    # and store the input for the backward pass
    self.X = X
    # note that X in the usual equation is a single vector
    # here is is a matrix made of a bunch of samples where axis 0 is the sample
    # and axis 1 is the dimension
    return self.X @ self.W + self.b

  
  def backward(self, dL_dW_out):
    # backward pass
    # dL_dW_out: N X M
    # N is the number of samples
    # M is the number of output dimensions
    # derivative of the loss with respect to weights of the output layer
    # self.W : wieghts of the input layer
    # rais errors if X is none
    if self.X is None:
      raise ValueError('X is None. Run forward pass first')
    
    # print(f'dL_dW_out: {dL_dW_out.shape}')

    # we want component wise multiplication
    # note the dl dw is a vector, but X is a batch of samples
    # print(f'X: {self.X.shape}')
    # print(f'dl dw out: {dL_dW_out.shape}')
    # X: (32, 256)
    # dl dw out: (32, 26)
    # dl_dw_in -> (32, 256), wont work other wise
    self.dL_dW_in = dL_dW_out[:,:,None] @ self.X[:,None,:]
    self.dL_db_in = dL_dW_out

    # print(f'dL_dX: {self.dL_dX.shape}')
    # print(f'dL_dW_in: {self.dL_dW_in.shape}')

    return np.dot(dL_dW_out, self.W.T)

class ReluLayer():
  
    def __init__(self):
      pass
  
    def forward(self, X):
      # forward pass
      # X: N X D
      # N is the number of samples
      # D is the number of dimensions
      # return the result of the forward pass
      # and store the input for the backward pass
      self.X = X
      # store gradient for backward pass
      self.dL_dX = d_relu(X)
      Z = relu(X)
      return Z
  
    def backward(self, dL_dX_out):
      # backward pass
      # dL_dX_out: N X D
      # N is the number of samples
      # D is the number of dimensions
      # derivative of the loss with respect to the input of the ReLU layer
      # self.X : input of the ReLU layer
      # raise errors if X is none
      if self.X is None:
        raise ValueError('X is None. Run forward pass first')
      # print(f'dL_dX_out: {dL_dX_out.shape}')
      # print(f'dL_dX: {self.dL_dX.shape}')
      return dL_dX_out * (self.X > 0)

class LeakyReluLayer():

  def __init__(self, alpha=0.2):
    self.alpha = alpha
  
  def forward(self, X):
    self.X = X
    self.dL_dX = np.where(X > 0, 1, self.alpha)
    return np.maximum(X, 0)
  
  def backward(self, dL_dX_out):
    return dL_dX_out * self.dL_dX

class SigmoidLayer():
    # internal sigmoid layer, not the final layer
  
    def __init__(self):
      pass
  
    def forward(self, X):
      self.X = X
      self.Y_hat = 1 / (1 + np.exp(-X))
      return self.Y_hat

    def backward(self, dL_dX_out):
      return dL_dX_out * self.Y_hat * (1 - self.Y_hat)


  

class SoftmaxLayer():
    
      def __init__(self):
        pass
    
      def forward(self, X):
        # forward pass
        # X: N X C
        # N is the number of samples
        # C is the number of classes
        # return the result of the forward pass
        # and store the input for the backward pass
        self.X = X
        self.Y_hat = softmax(X)
        return self.Y_hat
    
      def backward(self, Y):
        # backward pass
        # dL_dX_out: N X C
        # N is the number of samples
        # C is the number of classes
        # derivative of the loss with respect to the input of the softmax layer
        # self.X : input of the softmax layer
        # raise errors if X is none
        if self.X is None:
          raise ValueError('X is None. Run forward pass first')
        error = self.Y_hat - Y
        return error

class InputLayer():

  def __init__(self, input_shape):
    self.input_shape = input_shape

  def forward(self, X):
    return X

  def backward(self, Y):
    return Y


class AdamOptimizer():

  def __init__(self, lr, model, beta1=0.9, beta2=0.999, epsilon=1e-8):
    self.lr = lr
    self.model = model
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.m = [np.zeros_like(layer.W) for layer in model.layers if hasattr(layer, 'W')]
    self.v = [np.zeros_like(layer.W) for layer in model.layers if hasattr(layer, 'W')]
    self.t = 0
  
  def update(self, layer, lr):
    # update the weights and biases of the layer
    # layer: a layer object
    # lr: the learning rate
    # update the weights and biases of the layer
    # return the updated weights and biases
    self.t += 1
    idx = 0
    for layer in self.model.layers:
      if hasattr(layer, 'W'):
        self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * layer.dL_dW_in.mean(axis=0).T
        self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (layer.dL_dW_in.mean(axis=0).T ** 2)
        m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
        v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
        layer.W -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        idx += 1



class SGDOptimizer():

  def __init__(self, lr, model):
    self.lr = lr
    self.model = model

  def update(self, layer, lr):
    # update the weights and biases of the layer
    # layer: a layer object
    # lr: the learning rate
    # update the weights and biases of the layer
    # return the updated weights and biases
    layer.W -= lr * layer.dL_dW_in.mean(axis=0).T 
    layer.b -= lr * layer.dL_db_in.mean(axis=0).T
  
  def step(self):
    # update the weights and biases of each layer in the model
    # model: a model object
    # lr: the learning rate
    # update the weights and biases of each layer in the model
    # return the updated model
    for layer in self.model.layers:
      if hasattr(layer, 'dL_dW_in'):
        self.update(layer, self.lr)
    return self.model



def accuracy(Y, Y_hat):
  # calculate the accuracy
  # Y: N X C
  # Y_hat: N X C
  # N is the number of samples
  # C is the number of classes
  # return the accuracy
  return np.mean(np.argmax(Y, axis=1) == np.argmax(Y_hat, axis=1))



# MODEL

class Model():

  def __init__(self):
    self.layers = []
  
  def add(self, layer):
    self.layers.append(layer)
  
  def forward(self, X):
    # forward pass
    # X: N X D
    # N is the number of samples
    # D is the number of dimensions
    # run the forward pass of each layer
    # return the result of the forward pass
    for layer in self.layers:
      X = layer.forward(X)
    return X
  
  def backward(self, Y):
    # backward pass
    # Y: N X C
    # N is the number of samples
    # C is the number of classes
    # run the backward pass of each layer
    # return the derivative of the loss with respect to the input of the model
    dL_dX = Y
    for layer in reversed(self.layers):
      # print the layer type
      dL_dX = layer.backward(dL_dX)
    return dL_dX

  def compile(self, optimizer):
    self.optimizer = optimizer
  
  def fit(self, X_t, Y_t, X_v, Y_v, epochs, batch_size):
    # training loop
    # X_t: training features
    # Y_t: training labels
    # X_v: validation features
    # Y_v: validation labels
    # epochs: number of epochs
    # batch_size: batch size
    # run the training loop for the specified number of epochs
    # return the training history
    history = {
      'loss': [],
      'accuracy': [],
      'val_loss': [0],
      'val_accuracy': [0]
    }
    batches = range(0, X_t.shape[0], batch_size)
    pbar_epochs = tqdm(range(epochs))
    # add description with the loss and accuracy, and epoch number
    # update loss, accuracy and val_loss, val_accuracy after each epoch
    pbar_epochs.set_description(f'Epoch: {0}, Loss: {0}, Accuracy: {0} Val Loss: {0}, Val Accuracy: {0}')

    for epoch in pbar_epochs:
      pbar_batch = tqdm(batches, leave=False)
      pbar_batch.set_description(f'Batch: {0}, Loss: {0}, Accuracy: {0} Val Loss: {0}, Val Accuracy: {0}')
      for batch in pbar_batch:
        # get a random sample of indexes, of the batch size for the validation set
        # we want to sample the same porportion of the batch size to the training set
        p_v = batch_size / X_v.shape[0]
        val_indices = np.random.choice(X_v.shape[0], max(3, int(p_v * X_v.shape[0])), replace=False)
        val_Y_hat = self.forward(X_v[val_indices])
        val_loss = cross_entropy_loss(Y_v[val_indices], val_Y_hat)
        val_acc = accuracy(Y_v[val_indices], val_Y_hat)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        X_batch = X_t[batch:batch+batch_size]
        Y_batch = Y_t[batch:batch+batch_size]
        Y_hat = self.forward(X_batch)
        loss = cross_entropy_loss(Y_batch, Y_hat)
        acc = accuracy(Y_batch, Y_hat)
        self.backward(Y_batch)
        self.optimizer.step()
        history['loss'].append(loss)
        history['accuracy'].append(acc)
        curr_batch = batch // batch_size
        pbar_batch.set_description(f'Batch: {curr_batch}, Loss: {float(loss):4f}, Accuracy: {float(acc):4f}')
      
      pbar_epochs.set_description(f'Epoch: {epoch}, Loss: {float(loss):4f}, Accuracy: {float(acc):4f} Val Loss: {float(val_loss):4f}, Val Accuracy: {float(val_acc):4f}')
    
    return history
  
class FlatModel(Model):

  # this model is the same as the model class
  # but it can be initialized with a width and a depth, as well as an inputLayer and an outputLayer

  # note that
  
  def __init__(self, input_size, output_size, width, depth, ActivationLayer = ReluLayer):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.width = width
    self.depth = depth
    self.ActivationLayer = ActivationLayer

  def construct_layers(self):
    if self.depth == 0:
      self.add(InputLayer(self.input_size))
      self.add(DenseLayer(self.input_size, self.output_size))
      self.add(SoftmaxLayer())
    else:
      # note that if the hidden layer, is the first hidden layer, the input is the input size
      # if the hidden layer is the last hidden layer, the output size is the output size
      # otherwise its the width
      self.add(InputLayer(self.input_size))
      for i in range(self.depth):
        ith_layer_input_size = self.input_size if i == 0 else self.width
        self.add(DenseLayer(ith_layer_input_size, self.width))
        self.add(self.ActivationLayer())
      # there should always be one dense layer connected to the softmax layer
      # if there are no hidden layers, we have to connect the input layer to the softmax layer
      self.add(DenseLayer(self.width, self.output_size))
      self.add(SoftmaxLayer())
  

def experiment1(params, epochs, batch_size, lr, X_t, Y_t, X_v, Y_v, X_e, Y_e):
  # params: list of tuples of (width, depth)
  # run the experiment
  # return the results
  results = {}
  for param in params:
    width, depth = param
    model = FlatModel(X_t.shape[1], Y_t.shape[1], width, depth)
    model.construct_layers()
    optimizer = SGDOptimizer(lr, model)
    model.compile(optimizer)
    history = model.fit(X_t, Y_t, X_v, Y_v, epochs, batch_size)
    results[param] = {}
    results[param]['history'] = history
    results[param]['test_loss'] = cross_entropy_loss(Y_e, model.forward(X_e))
    results[param]['test_accuracy'] = accuracy(Y_e, model.forward(X_e))
    # plot the histry and save it
    histories = {
      'loss': [
        ('train', history['loss']),
        ('val', history['val_loss'])
      ],
      'accuracy': [
        ('train', history['accuracy']),
        ('val', history['val_accuracy'])
      ]
    }

    plot_history(histories, title=f'Width: {width}, Depth: {depth}', plot_file_path=f'figures/exp1/width_{depth}_{width}.png')
  
  return results

def experiment_1_all_models():
  model_params = []
  for i in range(0, 5):
    for j in range(3, 10):
      model_params.append((2**j, i))
  epochs = 3
  batch_size = 32
  lr = 0.1

  exp1_results = experiment1(model_params, epochs, batch_size, lr, X_t, Y_t, X_v, Y_v, X_e, Y_e)
  
  # plot the results where accuracy is on the y-axis and width is on the x-axis
  # there should be a line for each depth
  fig, ax = plt.subplots()

  # pull out the all the depths
  depths = set([depth for width, depth in model_params])

  for depth in depths:
    x = [width for width, depth_ in model_params if depth_ == depth]
    y = [exp1_results[(width, depth)]['test_accuracy'] for width in x]
    ax.plot(x, y, label=f'Depth: {depth}')
  ax.legend()
  # label the x and y axis
  ax.set_xlabel('Width')
  ax.set_ylabel('Accuracy')
  # save plot
  plt.savefig(f'figures/exp1/accuracy_vs_width.png')
  plt.close(fig)
  
  # print the results
  for key, value in exp1_results.items():
    print(f'Width: {key[0]}, Depth: {key[1]}, Test Loss: {value["test_loss"]}, Test Accuracy: {value["test_accuracy"]}')
  return exp1_results







# EXPERIMENT 2: Different activation functions
# -ReLu
# -Leaky ReLu
# -Sigmoid



def experiment2():

  model_params = []
  for i in range(1, 2):
    for j in range(1, 10):
      model_params.append((2**j, i))
  epochs = 3
  batch_size = 32
  lr = 0.1

  results = {}
  for activation in [ReluLayer, LeakyReluLayer, SigmoidLayer]:
    for param in model_params:
      width, depth = param
      model = FlatModel(X_t.shape[1], Y_t.shape[1], width, depth, activation)
      model.construct_layers()
      optimizer = SGDOptimizer(lr, model)
      model.compile(optimizer)
      history = model.fit(X_t, Y_t, X_v, Y_v, epochs, batch_size)
      results[(activation, param)] = {}
      results[(activation, param)]['history'] = history
      results[(activation, param)]['test_loss'] = cross_entropy_loss(Y_e, model.forward(X_e))
      results[(activation, param)]['test_accuracy'] = accuracy(Y_e, model.forward(X_e))
      # plot the histry and save it
      histories = {
        'loss': [
          ('train', history['loss']),
          ('val', history['val_loss'])
        ],
        'accuracy': [
          ('train', history['accuracy']),
          ('val', history['val_accuracy'])
        ]
      }

      plot_history(histories, title=f'Width: {width}, Depth: {depth}, Activation: {activation.__name__}', plot_file_path=f'figures/exp2/width_{depth}_{width}_{activation.__name__}.png')
  

  # plot the results where the x axis is width, and each line is a different activation function at a different depth
  fig, ax = plt.subplots()

  # pull out the all the depths
  depths = set([depth for width, depth in model_params])

  for depth in depths:
    for activation in [ReluLayer, LeakyReluLayer, SigmoidLayer]:
      x = [width for width, depth_ in model_params if depth_ == depth]
      y = [results[(activation, (width, depth))]['test_accuracy'] for width in x]
      ax.plot(x, y, label=f'Depth: {depth}, Activation: {activation.__name__}')
  
  ax.legend()
  # label the x and y axis
  ax.set_xlabel('Width')
  ax.set_ylabel('Accuracy')
  # save plot
  plt.savefig(f'figures/exp2/accuracy_vs_width.png')


  return results



def main():
  # exp1_results = experiment_1_all_models()
  exp2_results = experiment2()

if __name__ == '__main__':
  main()










      


  





















  




