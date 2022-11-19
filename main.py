import numpy as np


def generate_x_y():
    """
    This function generates the input and output data for the neural network.
    :return:
    - x: input data, a matrix of shape (25, 3) where each row is a training sample representing a 5x5 image of the
    digits 1, 2 and 3.
    - y: output data, a matrix of shape (3, 3) where each row is a one vector representing the result of the prediction
    first element is the probability of the input being 1 the second element is the probability of the input being 2 and
    so on.
    """
    matrixRepresentingImageOf1 = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ])
    matrixRepresentingImageOf2 = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ])
    matrixRepresentingImageOf3 = np.array([
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0],
    ])
    x = np.array([
        matrixRepresentingImageOf1.flatten(),
        matrixRepresentingImageOf2.flatten(),
        matrixRepresentingImageOf3.flatten(),
    ])

    y = np.array([[0], [1], [1], [0]])
    one = [1, 0, 0]
    two = [0, 1, 0]
    three = [0, 0, 1]

    y = np.array([one, two, three])
    return x, y


def generate_random_weights(nodes_in_input_layer,
                            nodes_in_output_layer,
                            nodes_in_hidden_layer):
    """
    This function generates random weights for the neural network.
    :param nodes_in_input_layer: The number of nodes in the input layer.
    :param nodes_in_output_layer: The number of nodes in the output layer.
    :param nodes_in_hidden_layer: The number of nodes in the hidden layer.
    :return:
    - W1: A matrix of weights for the input layer to the hidden layer.
    - W2: A matrix of weights for the hidden layer to the output layer.
    """
    W1 = np.random.rand(nodes_in_input_layer, nodes_in_hidden_layer)
    W2 = np.random.rand(nodes_in_hidden_layer, nodes_in_output_layer)
    return W1, W2


def generate_random_biases(nodes_in_hidden_layer, nodes_in_output_layer):
    """
    This function generates random biases for the neural network.
    :param nodes_in_hidden_layer: The number of nodes in the hidden layer.
    :param nodes_in_output_layer: The number of nodes in the output layer.
    :return:
    - b1: A vector of biases for the input layer to the hidden layer.
    - b2: A vector of biases for the hidden layer to the output layer.
    """
    b1 = np.random.rand(1, nodes_in_hidden_layer)
    b2 = np.random.rand(1, nodes_in_output_layer)
    return b1, b2


def sigmoid(x):
    """
    This function applies the sigmoid function to the input.
    :param x: The input of the sigmoid function.
    :return:
    - y: The output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    This function applies the derivative of the sigmoid function to the input.
    :param x: The input of the derivative of the sigmoid function.
    :return:
    - y: The output of the derivative of the sigmoid function.
    """
    return x * (1 - x)


def forward_propagation(x, W1, W2, b1, b2):
    """
    This function performs the forward propagation step of the neural network.
    :param x: The input data.
    :param W1: The weights for the input layer to the hidden layer.
    :param W2: The weights for the hidden layer to the output layer.
    :param b1: The biases for the input layer to the hidden layer.
    :param b2: The biases for the hidden layer to the output layer.
    :return:
    - a1: The output of the input layer, without applying the activation function.
    - z1: The output of the hidden layer, with the activation function applied.
    - a2: The output of the output layer, without applying the activation function.
    - y_pred: The prediction of the neural network. Aka the output of the output layer, with the activation function
    applied.
    """
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y_pred = sigmoid(a2)
    return a1, z1, a2, y_pred


def back_propagation(x, y, y_pred, z1, W2):
    """
    This function performs the back propagation step of the neural network.
    :param x: The input data.
    :param y: The output data.
    :param y_pred: The prediction of the neural network. Aka the output of the output layer, with the activation function
    :param z1: The output of the hidden layer, with the activation function applied.
    :param W2: The weights for the hidden layer to the output layer.
    :return:
    - dW1: The derivative of the cost function with respect to the weights for the input layer to the hidden layer.
    - dW2: The derivative of the cost function with respect to the weights for the hidden layer to the output layer.
    """
    dJdW2 = np.dot(z1.T, 2 * (y_pred - y) * sigmoid_derivative(y_pred))
    dJdW1 = np.dot(x.T, (np.dot(2 * (y_pred - y) * sigmoid_derivative(y_pred), W2.T) * sigmoid_derivative(z1)))
    return dJdW1, dJdW2


def gradient_descent(alpha, W1, W2, dJdW1, dJdW2):
    """
    This function performs the gradient descent step of the neural network.
    :param alpha: A scalar representing the learning rate.
    :param W1: A matrix of weights for the input layer to the hidden layer.
    :param W2: A matrix of weights for the hidden layer to the output layer.
    :param dJdW1: The derivative of the cost function with respect to the weights for the input layer to the hidden
    layer.
    :param dJdW2: The derivative of the cost function with respect to the weights for the hidden layer to the output
    layer.
    :return:
    - W1: The new matrix of weights for the input layer to the hidden layer.
    - W2: The new matrix of weights for the hidden layer to the output layer.
    """
    W1 = W1 - alpha * dJdW1
    W2 = W2 - alpha * dJdW2
    return W1, W2


def calculate_cost(y, y_pred):
    """
    This function calculates the cost of the neural network.
    :param y: The output data.
    :param y_pred: The prediction of the neural network. Aka the output of the output layer, with the activation function
    :return:
    - cost: The cost of the neural network as the mean squared error.
    """
    return np.mean(np.square(y_pred - y))


def main(alpha, epochs):
    """
    This function runs the neural network.
    :param alpha: The learning rate.
    :param epochs: The number of epochs aka the number of times the neural network will be trained.
    """
    nodes_in_input_layer = 25
    nodes_in_output_layer = 3
    nodes_in_hidden_layer = 5

    W1, W2 = generate_random_weights(nodes_in_input_layer,
                                     nodes_in_output_layer,
                                     nodes_in_hidden_layer)
    b1, b2 = generate_random_biases(nodes_in_hidden_layer, nodes_in_output_layer)

    x, y = generate_x_y()

    for i in range(epochs):
        a1, z1, a2, y_pred = forward_propagation(x, W1, W2, b1, b2)
        dJdW1, dJdW2 = back_propagation(x, y, y_pred, z1, W2)
        W1, W2 = gradient_descent(alpha, W1, W2, dJdW1, dJdW2)

        if i % 100 == 0:
            print("Loss function value: ", calculate_cost(y, y_pred))
            print("Predicted value: ", y_pred)
            print("Actual value: ", y)
            print("Iteration: ", i)
            print("===========================================")


if __name__ == "__main__":
    alpha = 0.05
    num_iterations = 20000

    main(alpha, num_iterations)
