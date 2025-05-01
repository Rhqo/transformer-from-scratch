import numpy as np

class Norm:
    def __init__(self, dim, eps=1e-5):
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))

class LayerNorm(Norm):
    def __init__(self, dim, eps=1e-5):
        super().__init__(dim, eps)

    def forward(self, x):
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        return out
    
    def __call__(self, x):
        return self.forward(x)

class BatchNorm(Norm):
    def __init__(self, dim, eps=1e-5):
        super().__init__(dim, eps)

    def forward(self, x):
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        return out
    

# x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# ln = LayerNorm(dim=3)
# print(ln.forward(x))
# bn = BatchNorm(dim=3)
# print(bn.forward(x))