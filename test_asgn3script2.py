
from asgn3script2 import *
# =====================TESTS=======================
import pytest as pt
import numpy as np

# set the numpy seed
np.random.seed(0)

X_e_copy = X_e.copy()
Y_e_copy = Y_e.copy()
X_e_pytest = X_e_copy[:200]
Y_e_pytest = Y_e_copy[:200]

X_t_pytest = X_t.copy()
Y_t_pytest = Y_t.copy()
Y_t_pytest = Y_t_pytest[:200]
X_t_pytest = X_t_pytest[:200]

@pt.fixture
def sample_X_e():
  return X_e_pytest

@pt.fixture
def sample_Y_e():
  return Y_e_pytest

@pt.fixture
def sample_Y_e_hat(sample_Y_e):
  # add some random noise to the sample_Y_e
  sample_Y_e_hat = sample_Y_e + np.random.normal(0, 1e-5, sample_Y_e.shape)
  # softmax the sample_Y_e_hat with tf
  sample_Y_e_hat = tf.nn.softmax(tf.cast(sample_Y_e_hat, tf.float32))
  # cast back to numpy
  sample_Y_e_hat = sample_Y_e_hat.numpy()
  return sample_Y_e_hat

@pt.fixture
def sample_X_t():
  return X_t_pytest

@pt.fixture
def sample_Y_t():
  return Y_t_pytest


def test_softmax_1(sample_X_e):
  # test the softmax function
  # we will use the tf implementation as the correct implementation
  # we will compare the output of the tf implementation with our implementation
  # we will use random values for the input

  tf_softmax = tf.nn.softmax(tf.cast(sample_X_e, tf.float32))
  custom_softmax = softmax(sample_X_e)
  np.testing.assert_allclose(tf_softmax, custom_softmax, rtol=1e-5, atol=1e-5)

def test_softmax_2(sample_X_e):
  # test softmax on very large random values
  # we will use the tf implementation as the correct implementation
  # check if there are any nan values
  # take sample_X_e and add 200x more rows with large values
  sample_X_e_large = np.vstack([sample_X_e]*500)
  # multiply the sample_X_e_large with a large number
  sample_X_e_large = sample_X_e_large * 1e5
  tf_softmax = tf.nn.softmax(tf.cast(sample_X_e_large, tf.float32))
  custom_softmax = softmax(sample_X_e_large)
  # check if there are any nan values
  assert not np.isnan(custom_softmax).any()
  np.testing.assert_allclose(tf_softmax, custom_softmax, rtol=1e-5, atol=1e-5)


def test_cross_entropy_loss_1(sample_Y_e, sample_Y_e_hat):
  # test the cross entropy loss function
  # we will use the tf implementation as the correct implementation
  # we will compare the output of the tf implementation with our implementation
  # we will use random values for the input

  tf_loss = tf.losses.categorical_crossentropy(tf.cast(sample_Y_e, tf.float32), tf.cast(sample_Y_e_hat, tf.float32))
  custom_loss = cross_entropy_loss(sample_Y_e, sample_Y_e_hat)
  np.testing.assert_allclose(tf_loss, custom_loss, rtol=1e-5, atol=1e-5)

def test_cross_entropy_loss3(sample_Y_e, sample_Y_e_hat):
  # set random sample_Y_e_hat values to 0
  sample_Y_e_hat = np.zeros_like(sample_Y_e_hat)

  # assert that a value error is raised
  with pt.raises(ValueError):
    cross_entropy_loss(sample_Y_e, sample_Y_e_hat)

def test_relu_1(sample_X_e):
  # test the relu function
  # we will use the tf implementation as the correct implementation
  # we will compare the output of the tf implementation with our implementation
  # we will use random values for the input

  tf_relu = tf.nn.relu(tf.cast(sample_X_e, tf.float32))
  custom_relu = relu(sample_X_e)
  np.testing.assert_allclose(tf_relu, custom_relu, rtol=1e-5, atol=1e-5)

def test_d_relu_1(sample_X_e):
  # test the relu derivative function
  # we will use the tf implementation as the correct implementation
  # we will compare the output of the tf implementation with our implementation
  # we will use random values for the input

  tf_d_relu = tf.cast(sample_X_e > 0, tf.float32)
  custom_d_relu = d_relu(sample_X_e)
  np.testing.assert_allclose(tf_d_relu, custom_d_relu, rtol=1e-5, atol=1e-5)


def test_denselayer_forward_pass_1(sample_X_e):

  # test the forward pass of the dense layer
  # we will use the tf implementation as the correct implementation
  # we will compare the output of the tf implementation with our implementation
  # we will use random values for the input

  # set the number of neurons in the layer
  input_size = sample_X_e.shape[1]
  output_size = 26
  # set the input size
  input_size = sample_X_e.shape[1]

  # create the weights and biases
  weights = np.random.normal(0, 1, (input_size, output_size))
  biases = np.random.normal(0, 1, (output_size))

  # forward pass
  tf_output = tf.matmul(tf.cast(sample_X_e, tf.float32), tf.cast(weights, tf.float32)) + tf.cast(biases, tf.float32)

  custom_output = DenseLayer(input_size, output_size, weights, biases).forward(sample_X_e)

  np.testing.assert_allclose(tf_output, custom_output, rtol=1e-3, atol=1e-2)


def test_dense_layer_backward():
    # Controlled test setup
    np.random.seed(0)
    sample_input = np.random.randn(2, 3)  # e.g., 2 samples, 3 features
    simulated_next_grad = np.random.randn(2, 4)  # gradient from next layer, 4 outputs

    # Initialize your DenseLayer
    dense_layer = DenseLayer(3, 4)  # 3 inputs, 4 outputs
    dense_layer.forward(sample_input)  # Forward pass
    computed_weight_gradient = dense_layer.backward(simulated_next_grad)
    computed_bias_gradient = dense_layer.dL_db_in

    # TensorFlow comparison
    # Create a new model with the same structure
    tf_input = tf.keras.Input(shape=(3,))
    tf_output = tf.keras.layers.Dense(4, use_bias=True)(tf_input)
    tf_model = tf.keras.Model(inputs=tf_input, outputs=tf_output)
    # Set TensorFlow model weights to DenseLayer weights
    tf_model.layers[1].set_weights([dense_layer.W, dense_layer.b])

    # Record gradient computation in TensorFlow
    with tf.GradientTape() as tape:
        tf_Y_hat = tf_model(sample_input)
        tf_loss = tf.reduce_mean(tf_Y_hat * simulated_next_grad)

    # Calculate gradients using TensorFlow
    tf_grads = tape.gradient(tf_loss, tf_model.trainable_weights)

    # Compare the gradients
    np.testing.assert_almost_equal(computed_weight_gradient, tf_grads[0].numpy(), decimal=5)
    np.testing.assert_almost_equal(computed_bias_gradient, tf_grads[1].numpy(), decimal=5)













