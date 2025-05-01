import numpy as np

class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.training = True

    def forward(self, x):
        if self.training:
            # 1: keep, 0: drop
            self.mask = np.random.binomial(
                1, 
                1 - self.dropout_rate, 
                size=x.shape
            )
            # inverted dropout
            return x * self.mask / (1 - self.dropout_rate)
        return x

    def __call__(self, x):
        return self.forward(x)
