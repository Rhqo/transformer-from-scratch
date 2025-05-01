import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim, bias=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.weights = np.random.randn(input_dim, output_dim) * 0.01

        if bias:
            self.bias_weights = np.random.randn(1, output_dim) * 0.01
        else:
            self.bias_weights = None

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        # y = x @ self.weights, matrix multiplication
        y = np.dot(x, self.weights)
        
        if self.bias:
            y += self.bias_weights
        return y
    
class SoftMax:
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=self.axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)