import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes=[11, 16, 16, 3], weights=None):
        self.layer_sizes = layer_sizes
        if weights is None:
            self.weights = []
            self.biases = []
            for i in range(len(layer_sizes) - 1):
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
                b = np.random.randn(layer_sizes[i+1]) * 0.5

                self.weights.append(w)
                self.biases.append(b)
        else:
            self.weights, self.biases = weights

    def forward(self, state):
        x = state.astype(np.float64)
        for i in range(len(self.weights)):
            x = x @ self.weights[i] + self.biases[i]
            if i < len(self.weights) - 1:
                x = np.tanh(x)
        return x

    def get_action(self, state):
        output = self.forward(state)
        idx = np.argmax(output)
        return ["straight", "left", "right"][idx]