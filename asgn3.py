
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

try:
    import cupy as cp
    use_cupy = False
except ImportError:
    import numpy as np
    use_cupy = False

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

# remove all but the first 2 classes
# X_e = X_e[np.argmax(Y_e, axis=1) < 2]
# Y_e = Y_e[np.argmax(Y_e, axis=1) < 2]
# X_t = X_t[np.argmax(Y_t, axis=1) < 2]
# Y_t = Y_t[np.argmax(Y_t, axis=1) < 2]

# display_images(X_e, Y_e, [0,1,2,3,4,5,6,7,8], 'Test Set')
# display_images(X_t, Y_t, [0,1,2,3,4,5,6,7,8], 'Train Set')


# X_e = standardize_data(X_e)
# X_t = standardize_data(X_t)

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
        # printe the type of w, and the type of x
        # print the type of b
        return (self.w[None, :, :] @ x[:, :, None]).squeeze() + self.b

    def backward(self, gradient):
        assert self.cur_input is not None, "Must call forward before backward"
        # If CuPy is not available, use NumPy arrays
        # Check if CuPy is available
        if use_cupy:
            # Convert gradient and cur_input to CuPy arrays if needed
            if not isinstance(gradient, cp.ndarray):
                gradient = cp.asarray(gradient)
            if not isinstance(self.cur_input, cp.ndarray):
                self.cur_input = cp.asarray(self.cur_input)
            
            # Compute gradients using CuPy arrays
            dw = gradient[:, :, None] @ self.cur_input[:, None, :]
            db = gradient.mean(axis=0)
            
            # Reshape db to have the same shape as dw_mean along the non-concatenating axis
            dw_mean = dw.mean(axis=0)
            db_reshaped = db.reshape(dw_mean.shape[0], -1)
            
            # Concatenate dw_mean and db_reshaped along the second axis
            self.gradient = cp.concatenate((dw_mean, db_reshaped), axis=1)
            
            # Compute the result using CuPy arrays
            result = gradient @ cp.asarray(self.w)
        else:
            # If CuPy is not available, use NumPy arrays
            dw = gradient[:, :, None] @ self.cur_input[:, None, :]
            db = gradient.mean(axis=0)
            self.gradient = [dw.mean(axis=0), db]
            result = gradient @ self.w

        return result

        # # Check if CuPy is available
        # if 'cupy' in globals():
        #     # Convert data to CuPy arrays
        #     gradient_cp = cp.asarray(gradient)
        #     cur_input_cp = cp.asarray(self.cur_input)
        #     w_cp = cp.asarray(self.w)

        #     # Perform operations using CuPy arrays
        #     dw_cp = gradient_cp[:, :, None] @ cur_input_cp[:, None, :]
        #     db_cp = gradient_cp.mean(axis=0)
        #     self.gradient = [cp.asnumpy(dw_cp.mean(axis=0)), cp.asnumpy(db_cp)]
        #     result_cp = gradient_cp @ w_cp

        #     # Convert the result back to NumPy array
        #     result = cp.asnumpy(result_cp)
        #     # print type of result
        #     print(f"Type of result: {type(result)}")
        # else:

    


class ReLULayer(NeuralNetLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        if use_cupy:
            self.gradient = cp.where(x > 0, 1.0, 0.0)
            return cp.maximum(0, x)
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
        if use_cupy:
            x_max = cp.max(x, axis=1, keepdims=True)
            expX = cp.exp(x - x_max)
            self.cur_probs = expX / expX.sum(axis=1, keepdims=True)
            return self.cur_probs
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
    
    def send_to_gpu(self):
        for layer in self.layers:
           # check if its a linear layer
           if isinstance(layer, LinearLayer):
               # send the weights to the gpu
               layer.w = cp.asarray(layer.w)
               layer.b = cp.asarray(layer.b)


               

    
    def send_to_cpu(self):
        for layer in self.layers:
            # check if its a linear layer
            if isinstance(layer, LinearLayer):
                # send the weights to the gpu
                layer.w = cp.asnumpy(layer.w)
                layer.b = cp.asnumpy(layer.b)

# %%
class Optimizer:
    def __init__(self, net: MLP):
        self.net = net

    def step(self):
        for layer in self.net.layers[::-1]:
            if layer.parameters is not None:
                self.update(layer.w, layer.b, layer.gradient[0], layer.gradient[1])

    def update(self, w, b, dw, db):
        raise NotImplementedError

class GradientDescentOptimizer(Optimizer):
    def __init__(self, net: MLP, lr: float):
        super().__init__(net)
        self.lr = lr

    def update(self, w, b, dw, db):
        w -= self.lr * dw.mean(axis=0)
        b -= self.lr * db.mean(axis=0)

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
    accuracies = []
    v_losses = []
    # labels = np.eye(3)[np.array(data_y)]
    labels = data_y

    # validation set 0.2% split of train
    # make sure its shuffled
    # shuffled_data_x = data_x.copy()
    # # shuffle only the rows
    # np.random.shuffle(shuffled_data_x)
    # split = int(len(shuffled_data_x) * 0.2)
    # validation_x = shuffled_data_x[:split]
    # validation_y = labels[:split]
    # data_x = shuffled_data_x[split:]
    # labels = labels[split:]

    validation_x = X_e.reshape(X_e.shape[0], -1)
    validation_y = Y_e

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

    if use_cupy:
        mlp.send_to_gpu()
        # send the data to the gpu
        data_x = cp.asarray(data_x)
        labels = cp.asarray(labels)
        validation_x = cp.asarray(validation_x)
        validation_y = cp.asarray(validation_y)
    
    # print the type of the weights for each layer


    pbar = tqdm(range(steps))
    for i in pbar:
        
        # if cuda is available, convert the the params in each layer to cupy arrays

        # loop over all samples
            # forward pass

        

        predictions = mlp.forward(data_x)



        loss = ce_loss(labels, predictions)
        # print the type of the loss
        
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

        # display loss in tqdm bar
        tqdm.set_description(pbar, f'Loss: {float(loss):.4f}, Accuracy: {float(accuracy):.4f}')
        losses.append(float(loss))
        accuracies.append(float(accuracy))
        mlp.backward(labels)
        optimizer.step()
        
        v_predictions = mlp.forward(validation_x)
        v_loss = ce_loss(validation_y, v_predictions)
        v_losses.append(float(v_loss))
    
    if use_cupy:
        mlp.send_to_cpu()
        data_x = cp.asnumpy(data_x)
        validation_x = cp.asnumpy(validation_x)
        validation_y = cp.asnumpy(validation_y)
    
    # Close the plot
    # plt.ioff()
    # plt.show()
    
    loss_dict = {
        'train': losses,
        'validation': v_losses,
        'accuracy': accuracies
    }
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



def model_accuracy(mlp, X_eval, Y_eval):
    
    if use_cupy:
        mlp.send_to_gpu()
        X_eval = cp.asarray(X_eval)
        Y_eval = cp.asarray(Y_eval)
    
    # reshape X_eval
    X_eval = X_eval.reshape(X_eval.shape[0], -1)
    print(f'X_eval shape: {X_eval.shape}')
    print(f'X peek: {X_eval[0:10]}')
    predictions = mlp.forward(X_eval)
    # print the first prediction, and the first label

    if use_cupy:
        mlp.send_to_cpu()
        X_eval = cp.asnumpy(X_eval)
        Y_eval = cp.asnumpy(Y_eval)
        predictions = cp.asnumpy(predictions)

    # save the photo of the first image to test_image.png
    # and display the correct label above the image
    # convert the one hot to the letter
    plt.imshow(X_eval[0].reshape(28,28), cmap='gray')
    predicted_letter = np.argmax(predictions[0])
    predicted_letter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[int(predicted_letter)]
    actual_letter = np.argmax(Y_eval[0])
    actual_letter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[int(actual_letter)]
    plt.title(f'Predicted: {predicted_letter} Actual: {actual_letter}')
    # save
    plt.savefig('test_image.png')

    # calculate the accuracy
    predicted_maxes = np.argmax(predictions, axis=1)
    actual_maxes = np.argmax(Y_eval, axis=1)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    print(f'Prediction\n {predictions[0:10]}')
    print(f'Actual\n {Y_eval[0:10]}')
    # print(f'Prediction{predicted_maxes[0]}')
    # print(f'Actual{actual_maxes[0]}')
    # print(f'Predicted type: {type(predicted_maxes)}')
    # print(f'Actual type: {type(actual_maxes)}')
    accuracy = np.mean(predicted_maxes == actual_maxes)
    print(f'Accuracy: {accuracy}')
    return accuracy

def save_loss_plot(loss_dict, filename):
    # clear plot of previous data
    plt.clf()
    for name, loss in loss_dict.items():
        plt.plot(loss, label=name)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def experiment1():
    
    # model params:
    model_param_tests = [(5,32)]
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


        # model accuracy pre training
        print('test accuracy pre training...')
        model_accuracy(mlp['model'], X_e, Y_e)

        # train accuracy
        print('train accuracy pre training...')
        model_accuracy(mlp['model'], X_t, Y_t)

        X_t_flat = X_t.reshape(X_t.shape[0], -1)
        LEARNING_RATE = 0.001
        mlp['losses'] = train(mlp['model'], GradientDescentOptimizer(mlp['model'], LEARNING_RATE), X_t_flat, Y_t, 10)
        # calculate the accuracy test and train accuracy
        # evaluation accuracy
        print('Evaluating model...')
        model_accuracy(mlp['model'], X_e, Y_e)

        print('Training accuracy...')
        model_accuracy(mlp['model'], X_t, Y_t)

        # eval_acc = np.mean(np.argmax(mlp['model'].forward(X_e.reshape(X_e.shape[0], -1)), axis=1) == np.argmax(Y_e, axis=1))
        # print(f'Eval Accuracy: {eval_acc}')
        # mlp['eval_acc'] = eval_acc
        # train_acc = np.mean(np.argmax(mlp['model'].forward(X_t_flat), axis=1) == np.argmax(Y_t, axis=1))
        # print(f'Train Accuracy: {train_acc}')
        # mlp['train_acc'] = train_acc

        # save the loss plot
        import datetime

        # convert the losses to numpy arrays
        # print the type of the losses
        for k, v in mlp['losses'].items():
            print(f'Type of {k}: {type(v)}')

        save_loss_plot(mlp['losses'], f'loss_plot_{datetime.datetime.now()}.png')
    
    return MLPs


MLPs = experiment1()

for params, mlp in MLPs.items():
    print(f'Model: {params}')


# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam


# def experiment1():
    

    
#     # model params:
#     model_param_tests = [(4,256)]

#     def get_tf_model(depth, width):
#         model = Sequential()
#         model.add(Dense(width, activation='relu', input_shape=(784,)))
#         for _ in range(depth - 2):
#             model.add(Dense(width, activation='relu'))
#         model.add(Dense(24, activation='softmax'))
#         model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#         return model
    
#     MLPs = { params : {'model':get_tf_model(*params)} for params in model_param_tests}



#     for mlp in MLPs.values():

#         X_t_flat = X_t.reshape(X_t.shape[0], -1)
#         history = mlp['model'].fit(X_t_flat, Y_t, epochs=20, batch_size=32)

#         # calculate the accuracy test and train accuracy
#         # evaluation accuracy
#         print('Evaluating model...')
#         test_loss, test_acc = mlp['model'].evaluate(X_e.reshape(X_e.shape[0], -1), Y_e)
#         print(f'Test Accuracy: {test_acc}')

#         mlp['losses'] = {
#             'train': history.history['loss'],
#             # 'validation': history.history['val_loss']
#         }

#         import datetime

#         # convert the losses to numpy arrays
#         # print the type of the losses
#         for k, v in mlp['losses'].items():
#             print(f'Type of {k}: {type(v)}')

#         save_loss_plot(mlp['losses'], f'loss_plot_{datetime.datetime.now()}.png')
    
#     return MLPs


# MLPs = experiment1()

# for params, mlp in MLPs.items():
#     print(f'Model: {params}')
#     # print(f'Eval Accuracy: {mlp["eval_acc"]}')
#     # print(f'Train Accuracy: {mlp["train_acc"]}')







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


