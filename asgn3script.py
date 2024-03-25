
# %% [markdown]
# # Assignment 3: Classification of Image Data
# - COMP 551 Winter 2024, Mcgill University
# - Rudolf Kischer: 260956107
# 

# %% [markdown]
# ### Synopsis
# - In this miniproject, we will implement a multilayer perceptron from scratch, and use it to classify image data. The goal is to implement a basic neural network and its training algorithm from scratch and get hands-on experience with important decisions that you have to make while training these models. You will also have a chance to experiment with convolutional neural networks.

# %% [markdown]
# # Data
# 
# - [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data)
# - Features: 28 x 28 grayscale images of hand symbols (784 pixels/features)
# - Labels: 24 classes of letters (excludes 9=J and 25=Z because they require motion)
# - Train: 27,455
# - Test: 7172 
# - Most of the images are produced through an alteration of 1704 uncropped color images
# - These alternations include the following:
#     - "To create new data, an image pipeline was used based on ImageMagick and included cropping to hands-only, gray-scaling, resizing, and then creating at least 50+ variations to enlarge the quantity. The modification and expansion strategy was filters ('Mitchell', 'Robidoux', 'Catrom', 'Spline', 'Hermite'), along with 5% random pixelation, +/- 15% brightness/contrast, and finally 3 degrees rotation."
# - CSV format, (label, pixel1, pixel2, ... , pixel784)
# - <img width=400 src="https://storage.googleapis.com/kagglesdsdata/datasets/3258/5337/amer_sign3.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com/20240314/auto/storage/goog4_request&X-Goog-Date=20240314T192335Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=5090d6842cb28ba5080a37a44706cfab4cba880c104d7acf1510df2a187f3c644bccde7d786cf964d8704f172a1b288bff914ae767deace400edfbd0610023d7cc6c6e329c2d365dc9f5a81c6bfe641800d6c7ecb500470fb48cabf2b555080be0f07559522be5487e6f3f456e8c20b909a818ffd6eaf2658089c82659443e1df42d0c06956fd5f46d9d1b9dfd6458ab03e47796b278463a2d1ebbeac2328b7ba668662807ce3b138e72afca7e9f29d4d01854d0ed4e8416afc4206787976e861cc0f14d9755542f06ee1a52e71e16a112f7e2e1e53a6136d711f54a64e8ad531c07083108fd034a1bf8cf04a5c9a13f94ca6fa0291fb1c60dc9b7629095b1a9"/>
# - <img width=400 src="https://storage.googleapis.com/kagglesdsdata/datasets/3258/5337/american_sign_language.PNG?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com/20240315/auto/storage/goog4_request&X-Goog-Date=20240315T200917Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=208afda814e246ab63f6d5f39436a53bcb0aa69be2ca78290e4caf450a4b47f8ea3fd9686815a4836e37c35682e0cc70c122f52f664ee18fa36407caa91f3f00b3874aa926ca4fab2d2366e114da10cad6014c1e8d978a80a150c45e3b4b5a855756a5d8ac9e1c606674728b5868a48e954329c9b41af9a3a0b912fedecf4d2bca40407add7f87d4c4bd57a423dbb4257b73fe0bf5830b81eadea549a41dd70b47c9acc9150078416f517b2814578506b379aee8543fe99f8e060ac978dd21dfc9dab5d7702a1d16b9ccc330500f9204e43ca21462f6bb3f48a70a70c7445f88a0a02e96a587c4babb965eb12adc6b2ca82e3cd6e91026f5204e93edc8bb82f2"/>
# 

# %% [markdown]
# # Setup

# %%


# %%
# IMPORTS

import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm

# %% [markdown]
# ## Data Processing
# - To prepare the data for usage we need to:
#   - download the dataset
#   - load the dataset
#   - seperate into X and Y
#   - vectorize the image data
#   - Center and normalize the data

# %%

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
  X = df.drop(['label'],axis=1)
  X = X.values.reshape(-1,28,28,1)
  # one hot encode

  # print the shape of Y
  print(f'Y shape: {Y.shape}')
  
  return X, Y

def standardize_data(X):
  # center by subtracting the mean
  # divide by the standard deviation
  # N X D 
  # N X W X H X C
  # copy the data
  X = X.copy()
  X = X - X.mean(axis=0)
  X = X / X.std(axis=0)
  return X

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

  

test, train = load_mnist_sign_dataset()
X_e, Y_e = shape(test)
X_t, Y_t = shape(train)

# display_images(X_e, Y_e, [0,1,2,3,4,5,6,7,8], 'Test Set')
# display_images(X_t, Y_t, [0,1,2,3,4,5,6,7,8], 'Train Set')


# X_e = standardize_data(X_e)
X_t = standardize_data(X_t)

#display first 5 images
# display_images(X_e, Y_e, [0,1,2,3,4,5,6,7,8], 'Test Set Standardized')
# display_images(X_t, Y_t, [0,1,2,3,4,5,6,7,8], 'Train Set Standardized')

# print the dimensions
print(f'Test Set: X:{X_e.shape} Y:{Y_e.shape}')
print(f'Train Set: X:{X_t.shape} Y:{Y_t.shape}')




# %% [markdown]
# # Model
# - We will be implementing a deep multi layered perceptron
# - We want it to be custimizable for different depths and breadths as well as activation functions for experimentations
# - We will also want to implement convolutional layers as well to test the performance increase
# - Each model will be composed of multiple layers, each which has a forward and backward function to update the weights of that layer
# - [MLP code Example](https://colab.research.google.com/github/yueliyl/comp551-notebooks/blob/master/MLP.ipynb)
# - [Deep MLP code Example](https://colab.research.google.com/github/yueliyl/comp551-notebooks/blob/master/NumpyDeepMLP.ipynb)
# 
# 

# %% [markdown]
# #### Activation functions

# %%
# Here a_j is the the output of the pre-activate hidden unit Zl_k
# where l is the lth layer and k is the kth unit in the lth layer
# A_l is an K_l x 1 vector where K_l is the number of units in the lth layer
# A_l is the pre-activated output of the lth layer
# Z_l is the post activated output of the lth layer
# Z_l = phi_l(A_l) where phi_l is the activation function of the lth layer

def relu(A_l):
  return np.maximum(0, A_l)

def heaviside(A_l):
  return np.heaviside(A_l, 0)

def sigmoid(A_l):
  return 1 / (1 + np.exp(-A_l))

def leaky_relu(A_l, alpha=0.01):
  return np.maximum(alpha * A_l, A_l)

def swish(A_l, beta=1):
  return A_l / (1 + np.exp(-beta * A_l))

def softmax(A_l):
  expA = np.exp(A_l)
  return expA / expA.sum(axis=1, keepdims=True)

# %% [markdown]
# ### Gradients

# %%
# A_l is the pre-activated output of the lth layer
# it is a K_l x 1 vector, so these derivatives output a gradient

# we want to ouput a vector of the same shape as A_l
def d_relu(A_l):
  return np.heaviside(A_l, 0)

def d_heaviside(A_l):
  return np.heaviside(A_l, 0)

def d_sigmoid(A_l):
  return A_l * (1 - A_l)

def d_leaky_relu(A_l, alpha=0.01):
  return np.maximum(alpha, np.heaviside(A_l, 0))

def d_swish(A_l, beta=1):
  return swish(A_l, beta) + beta * A_l * (1 - swish(A_l, beta))

def d_numerical(A_l, phi, h=1e-5):
  return (phi(A_l + h) - phi(A_l - h)) / (2 * h)





# %% [markdown]
# ### Loss Functions

# %%

def cross_entropy(Y, Y_hat):
  # -1 dot Y (*) log(Y_hat)
  # (*) element wise multiplication
  # Y is the output layer activations of the nework
  # Y_hat is the predicted output layer activations
  CE = -np.ones(Y.shape).dot(Y * np.log(Y_hat))
  return CE

def mean_squared_error(Y, Y_hat):
  return np.mean((Y - Y_hat)**2)

def d_cross_entropy(Y, Y_hat):
  # -Y / Y_hat
  return -Y / Y_hat

def d_mean_squared_error(Y, Y_hat):
  return Y - Y_hat



# %% [markdown]
# ### Layer

# %%
class NeuralNetLayer:
    def __init__(self):
        self.gradient = None
        self.parameters = None
        
    def forward(self, x):
        raise NotImplementedError

    def backward(self, gradient):
        raise NotImplementedError


class LinearLayer(NeuralNetLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.ni = input_size
        self.no = output_size
        self.w = np.random.randn(output_size, input_size)
        self.b = np.random.randn(output_size)
        self.cur_input = None
        self.parameters = [self.w, self.b]

    def forward(self, x):
        self.cur_input = x
        # n is the number of units in the previous layer
        # m is the number of units in the next layer
        # x_i = n x 1
        # W : (m x n)
        # x : (p x n)
        # W_ : (1 x m x n)
        # x_ : (p x n x 1)
        # W_ @ x_ : (1 x m x n) @ (p x n x 1) -> broadcast-> (p x m x n) @ (p x n x 1) = (p x m x 1) -> [(mx1) ...<-p->... (mx1)]
        # = (p x m x 1).squeeze() -> (p x m)
        # (W_ @ x_) + b = (p x m) + (1 x m) -> brodcast -> (p x m) + (p x m) = (p x m) )
        # print(f'w shape: {self.w.shape}')
        # print(f'w: {self.w}')
        return (self.w[None, :, :] @ x[:, :, None]).squeeze() + self.b

    def backward(self, gradient):
        assert self.cur_input is not None, "Must call forward before backward"
        #dw = gradient.dot(self.cur_input)
        # current input p x 1 x n
        # gradient p x m x 1
        dw = gradient[:, :, None] @ self.cur_input[:, None, :]
        db = gradient.mean(axis=0)
        self.gradient = [dw.mean(axis=0), db]
        return gradient @ self.w
    


class ReLULayer(NeuralNetLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        self.gradient = np.where(x > 0, 1.0, 0.0)
        return np.maximum(0, x)

    def backward(self, gradient):
        assert self.gradient is not None, "Must call forward before backward"
        return gradient * self.gradient
    
class SoftmaxOutputLayer(NeuralNetLayer):
    def __init__(self):
        super().__init__()
        self.cur_probs = None

    def forward(self, x):
        # Subtract the maximum value in each row for numerical stability
        x_max = np.max(x, axis=1, keepdims=True)
        expX = np.exp(x - x_max)
        
        # Compute softmax probabilities
        self.cur_probs = expX / expX.sum(axis=1, keepdims=True)
        return self.cur_probs

    def backward(self, target):
        assert self.cur_probs is not None, "Must call forward before backward"
        return self.cur_probs - target


# %% [markdown]
# ### Network

# %%


# %%

from typing import List
class MLP:
    def __init__(self, *args: List[NeuralNetLayer]):
        self.layers = args

    def forward(self, x):
        # print(x.shape)
        
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, target):
        for layer in self.layers[::-1]:
            target = layer.backward(target)

# %%
class Optimizer:
    def __init__(self, net: MLP):
        self.net = net

    def step(self):
        for layer in self.net.layers[::-1]:
            if layer.parameters is not None:
                self.update(layer.parameters, layer.gradient)

    def update(self, params, gradient):
        raise NotImplementedError

class GradientDescentOptimizer(Optimizer):
    def __init__(self, net: MLP, lr: float):
        super().__init__(net)
        self.lr = lr

    def update(self, params, gradient):
        for (p, g) in zip(params, gradient):
            p -= self.lr * g.mean(axis=0)

# %%

def ce_loss(y, y_hat):
    epsilon = 1e-10
    return -(y * np.log(y_hat + epsilon)).sum(axis=-1).mean()

def plot_losses(loss_dict):
    for name, loss in loss_dict.items():
        plt.plot(loss, label=name)
    plt.legend()
    plt.show()

from IPython.display import clear_output, display

def train(mlp: MLP, optimizer: Optimizer, data_x, data_y, steps):
    losses = []
    v_losses = []
    # labels = np.eye(3)[np.array(data_y)]
    labels = data_y

    # validation set 0.2% split of train
    # make sure its shuffled
    shuffled_data_x = data_x.copy()
    # shuffle only the rows
    np.random.shuffle(shuffled_data_x)
    split = int(len(shuffled_data_x) * 0.2)
    validation_x = shuffled_data_x[:split]
    validation_y = labels[:split]
    data_x = shuffled_data_x[split:]
    labels = labels[split:]

    # Create a figure and axes for plotting
    # fig, ax = plt.subplots()
    # line1, = ax.plot([], [], 'b-', label='Training Loss')
    # line2, = ax.plot([], [], 'r-', label='Validation Loss')
    # ax.set_xlabel('Iteration')
    # ax.set_ylabel('Loss')
    # ax.set_title('Training and Validation Loss')
    # ax.legend()

    # Initialize the plot
    # plt.ion()
    # plt.show()

    batch_size = 32
    batches = len(data_x) // batch_size

    pbar = tqdm(range(steps))
    pbar_outer = tqdm(range(batches))
    for i in pbar:
        
        for j in pbar_outer:
            # Get the current batch
            start = j * batch_size
            end = start + batch_size
            batch_x = data_x[start:end]
            batch_y = labels[start:end]

            predictions = mlp.forward(batch_x)
            np.set_printoptions(precision=3)
            np.set_printoptions(suppress=True)
            loss = ce_loss(batch_y, predictions)

            # display loss in tqdm bar
            # print(f'Loss: {loss}')
            losses.append(loss)
            mlp.backward(batch_y)
            optimizer.step()
            
            v_predictions = mlp.forward(validation_x)
            v_loss = ce_loss(validation_y, v_predictions)
            v_losses.append(v_loss)

            tqdm.set_description(pbar, f'Loss: {loss:.5f} Validation Loss: {v_loss:.5f}')
    
    # Close the plot
    # plt.ioff()
    # plt.show()
    
    loss_dict = {
        'train': losses,
        'validation': v_losses
    }
    # plot_losses(loss_dict)
    return loss_dict

# %% [markdown]
# # Experiments

# %% [markdown]
# ### Exp. 1: MLP Layer Depth
# - models:
#   - no hidden layers
#   - single hidden layer, ReLU
#   - two hidden layers, Relu
# - for each experiment with
#   - 32, 64, 128, 256
# - Output layer is softmax
# - compare test accuracy
# - comment on non-linearity

# %%
def get_model(depth, width, input_size, output_size):
    layers = []
    for i in range(depth - 1):
        # if this is the last layer, we need to set the output size instead
        # if this is the first layer we need to set the input size to the input size
        layer_input_size = width if i > 0 else input_size
        layer_output_size = width if i < depth - 2 else output_size
        layers.append(LinearLayer(layer_input_size, layer_output_size))
        if i < depth - 2:
            layers.append(ReLULayer())

    

    layers.append(SoftmaxOutputLayer())
    return MLP(*layers)



def experiment1():
    
    # model params:
    model_param_tests = [(4,512)]
    # for depth in range(3, 5):
    #     for width in [32, 64, 128, 256]:
    #         model_param_tests.append((depth, width))


    MLPs = { params : {'model':get_model(*params, 784, 24)} for params in model_param_tests}
    # MLPs = [get_model(*params, 784, 24) for params in model_param_tests]

    for mlp in MLPs.values():
        # make the training data linear
        # x shape: (27455, 28, 28, 1)
        # we need to flatten the data
        # x_flat shape: (27455, 784)
        X_t_flat = X_t.reshape(X_t.shape[0], -1)
        mlp['losses'] = train(mlp['model'], GradientDescentOptimizer(mlp['model'], 0.0001), X_t_flat, Y_t, 10)
        # calculate the accuracy test and train accuracy
        # evaluation accuracy
        print('Evaluating model...')
        eval_acc = np.mean(np.argmax(mlp['model'].forward(X_e.reshape(X_e.shape[0], -1)), axis=1) == np.argmax(Y_e, axis=1))
        print(f'Eval Accuracy: {eval_acc}')
        mlp['eval_acc'] = eval_acc
        train_acc = np.mean(np.argmax(mlp['model'].forward(X_t_flat), axis=1) == np.argmax(Y_t, axis=1))
        print(f'Train Accuracy: {train_acc}')
        mlp['train_acc'] = train_acc
    
    return MLPs


MLPs = experiment1()

for params, mlp in MLPs.items():
    print(f'Model: {params}')
    print(f'Eval Accuracy: {mlp["eval_acc"]}')
    print(f'Train Accuracy: {mlp["train_acc"]}')







# %%


# %% [markdown]
# ### Exp. 2: MLP activation Function

# %% [markdown]
# ### Exp. 3: Regularization

# %% [markdown]
# ### Exp. 4: Convultional Neural Net

# %% [markdown]
# ### Exp. 5: Optimizing MLP Architecture

# %% [markdown]
# ### Exp. 6: Report Results

# %% [markdown]
# # Experiment Extensions

# %% [markdown]
# # Results And Conclusion


