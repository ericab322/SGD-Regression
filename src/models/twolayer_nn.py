import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# try iid gaussians with mean zero and variance 1
class TwoLayerNNModel:
    def __init__(self, input_dim, hidden_dim=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1 

    def initialize_weights(self):
        limit_W1 = np.sqrt(2 / self.input_dim)
        limit_W2 = np.sqrt(2 / self.hidden_dim)

        self.W1 = np.random.randn(self.hidden_dim, self.input_dim) * limit_W1
        self.b1 = np.zeros((self.hidden_dim, 1))
        self.W2 = np.random.randn(self.output_dim, self.hidden_dim) * limit_W2
        self.b2 = np.zeros((self.output_dim, 1))

        return self.pack_weights(self.W1, self.b1, self.W2, self.b2)

    def pack_weights(self, W1, b1, W2, b2):
        return np.concatenate([W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()])

    def unpack_weights(self, w):
        d, h, o = self.input_dim, self.hidden_dim, self.output_dim
        i = 0
        W1 = w[i:i + h * d].reshape(h, d)
        i += h * d
        b1 = w[i:i + h].reshape(h, 1)
        i += h
        W2 = w[i:i + o * h].reshape(o, h)
        i += o * h
        b2 = w[i:i + o].reshape(o, 1)
        return W1, b1, W2, b2

    def forward(self, x, w):
        W1, b1, W2, b2 = self.unpack_weights(w)
        z0 = x.reshape(-1, 1)
        h1 = W1 @ z0 + b1
        z1 = relu(h1)
        h2 = W2 @ z1 + b2
        return h2.item()
    
    def backprop(self, w, x, y):
        W1, b1, W2, b2 = self.unpack_weights(w)
        x = x.reshape(-1, 1)

        h1 = W1 @ x + b1
        z1 = relu(h1)
        h2 = W2 @ z1 + b2
        pred = h2.item()

        error = 2 * (pred - y)
        dW2 = error * z1.T
        db2 = np.array([[error]])

        D1 = relu_derivative(h1)
        e1 = (W2.T * error) * D1
        dW1 = e1 @ x.T
        db1 = e1

        return np.concatenate([dW1.flatten(), db1.flatten(), dW2.flatten(), db2.flatten()])

    def F(self, X, y, w):
        # loss function
        preds = np.array([self.forward(x, w) for x in X])
        return np.mean((preds - y.flatten()) ** 2)

    def grad_F(self, X, y, w):
        # gradient of the loss function
        grads = [self.backprop(w, X[i], y[i]) for i in range(len(X))]
        return np.mean(grads, axis=0)

    def stochastic_grad(self, X, y, w):
        i = np.random.randint(len(X))
        return self.backprop(w, X[i], y[i])

    def mini_batch_grad(self, X, y, w, batch_size):
        idxs = np.random.choice(len(X), batch_size, replace=False)
        grads = [self.backprop(w, X[i], y[i]) for i in idxs]
        return np.mean(grads, axis=0)
