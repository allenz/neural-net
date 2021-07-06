# Covers standard layers, activations, and losses, including all formulations
# of the cross entropy loss. For an overview of cross entropy losses and
# gradient calculations, see
# https://allenzhu.com/notes/logistic_regression.html.

import numpy as np


class Module:
    # A Module implements a differentiable operation.
    # Subclasses must implement forward() and backward().
    # For trainable layers, I use a naming convention where the field
    # grad_weight stores the gradient for the weight parameter.
    
    def __call__(self, *args):
        return self.forward(*args)

    def __repr__(self):
        return type(self).__name__ + '()'

    def parameters(self):
        # Returns the trainable parameters for this module as a list of
        # (param, grad) pairs.
        params = []
        for attr, value in self.__dict__.items():
            if attr.startswith('grad_'):
                grad = value
                param_name = attr[5:]
                param = getattr(self, param_name)
                params.append((param, grad))
            elif isinstance(value, Module):
                params += value.parameters()
            elif attr == 'modules':
                params += [p for module in value for p in module.parameters()]
        return params


class Linear(Module):
    # Applies a linear transformation with learnable weight and bias.
    # Does not apply any activation.

    def __init__(self, d_in, d_out, activation_gain=1):
        # Initializes weights from a normal distribution so that the outputs
        # and gradients have approximately unit gain. Derivation at
        # https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init.
        # activation_gain specifies the gain needed to offset a subsequent
        # activation function. For example, ReLU activations will halve output
        # variance. PyTorch doesn't have this parameter so in PyTorch it's
        # tedious to initialize weights correctly.

        gain = (2/(d_in+d_out))**0.5

        self.w = gain*activation_gain*np.random.randn(d_in, d_out).astype('f')
        self.b = np.zeros(d_out, dtype=np.float32)
        self.grad_w = np.zeros((d_in, d_out))
        self.grad_b = np.zeros((d_out,))

    def forward(self, x):
        # x (B, d_in), returns (B, d_out)
        self.x = x
        return x @ self.w + self.b

    def backward(self, grad):
        # grad (B, d_out), returns (B, d_in)
        self.grad_w[:] += self.x.T @ grad / len(grad)
        self.grad_b[:] += grad.mean(axis=0)
        return grad @ self.w.T


def Dense(d_in, d_out, activation):
    # Applies a linear transformation and then an element-wise activation
    
    assert type(activation) == type
    a_gain = {None: 1, Tanh: 1, ReLU: 2**0.5, Sigmoid: 4}
    return Sequential(Linear(d_in, d_out, a_gain[activation]), activation())


# Binary logistic regression
class Sigmoid(Module):
    # Converts logits (B, 1) to probabilities

    def forward(self, logits):
        self.y_hat = 1 / (1 + np.exp(-logits))
        return self.y_hat
    
    def backward(self, grad):
        return grad * self.y_hat * (1 - self.y_hat)

Logistic = Sigmoid # aka expit


class LogSigmoid(Module):
    # Converts logits (B, 1) to log probabilities

    def forward(self, logits):
        self.logits = logits
        return -np.logaddexp(0, -logits)

    def backward(self, grad=1):
        return grad / (1 + np.exp(self.logits))


class BCELoss(Module):
    # Takes predicted probs y_hat and true probs/classes y (B, 1).
    # Applies the loss elementwise and then computes the mean.
    # The minimum for this loss is the entropy of the distribution of y.
    # The maximum entropy of a binary distribution is log(2) = 0.693.

    def forward(self, y_hat, y, eps=1e-8):
        assert all(arr.all() for arr in [0<=y, y<=1, 0<=y_hat, y_hat<=1])
        self.y_hat = y_hat
        self.y = y
        return -(y*np.log(y_hat+eps)+(1-y)*np.log(1-y_hat+eps)).mean()
    
    def backward(self, grad=1, eps=1e-16):
        return grad * (-self.y/(self.y_hat+eps) +
                       (1-self.y)/(1-self.y_hat+eps))/len(self.y)


class BCEWithLogitsLoss(Module):
    # Combines Sigmoid and BCELoss.

    def forward(self, logits, y):
        self.y_hat = 1 / (1 + np.exp(-logits))
        self.y = y
        return (np.logaddexp(0, -logits) + logits*(1-y)).mean()
        # more stable than (np.log(1+np.exp(-logits)) + logits*(1-y)).mean()

    def backward(self, grad=1):
        return grad * (self.y_hat - self.y)/len(self.y)


# Multinomial logistic regression
class Softmax(Module):
    # Converts scores (B, m) to probabilities (B, m).
    # Each output row represents a categorical distribution over m classes

    def forward(self, scores):
        e = np.exp(scores - scores.max())
        self.probs = e/e.sum(axis=-1, keepdims=True)
        return self.probs

    def backward(self, grad=1):
        pg = (self.probs * grad).sum(axis=-1, keepdims=True)
        return self.probs * (grad - pg)


class LogSoftmax(Module):
    # Applies log(softmax()), producing log probabilities

    def forward(self, scores):
        # division becomes subtraction in log space
        self.log_probs = scores - np.logaddexp.reduce(scores, axis=-1,
                                                      keepdims=True)
        return self.log_probs

    def backward(self, grad=1):
        # Derivation at https://stackoverflow.com/a/35328031
        return grad - np.exp(self.log_probs)*grad.sum(axis=-1, keepdims=True)


class NLLLoss(Module):
    # Takes log_probs (B, m) and true classes y (B,), returns scalar

    def forward(self, log_probs, y):
        # max P(y) = max prod P_i[y_i] = max sum log P_i[y_i]
        self.shape = log_probs.shape
        self.y = y
        return -log_probs[range(len(y)), y].mean()

    def backward(self, grad=1):
        g = np.zeros(self.shape)
        g[range(len(self.y)), self.y] = -1
        return grad * g / len(self.y)


class CrossEntropyLoss(Module):
    # Takes scores (B, m) and true classes y (B,), returns scalar
    # Computes NLLLoss(LogSoftmax(scores), y)

    def forward(self, scores, y):
        self.log_probs = LogSoftmax()(scores)
        self.y = y
        return -self.log_probs[range(len(y)), y].mean()

    def backward(self, grad=1):
        # Derivation at https://deepnotes.io/softmax-crossentropy
        g = np.exp(self.log_probs)
        g[range(len(self.y)), self.y] -= 1
        return grad * g / len(self.y)


# Activations
class ReLU(Module):
    # Popular nonlinearity without gradient saturation

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad):
        out = grad.copy()
        out[self.x < 0] = 0
        return out


class Tanh(Module):
    # Performs a bit better than sigmoid for multilayer networks, but still
    # suffers from gradient saturation. Maps real inputs to the range (-1, 1).

    def forward(self, x):
        self.t = np.tanh(x)
        return self.t

    def backward(self, grad):
        return grad * (1 - self.t**2)


class LeakyReLU(Module):
    # Add a small slope for negative inputs to address the dying ReLU
    # problem. Sometimes used for GANs.

    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        self.x = x
        return np.maximum(x, x * self.slope)

    def backward(self, grad):
        out = np.copy(grad)
        out[self.x < 0] *= self.slope
        return out


# Losses
class Mean(Module):
    # This layer is implicit when computing the mean of a decomposable loss

    def forward(self, x):
        # x: vector or matrix, returns a scalar
        self.shape = x.shape
        return x.mean()

    def backward(self, grad=1):
        # grad: scalar, returns a vector or matrix
        return grad * np.ones(self.shape) / np.prod(self.shape)


class MSELoss(Module):
    def forward(self, y_hat, y):
        self.delta = y_hat - y
        return (self.delta**2).mean()

    def backward(self, grad=1):
        return grad * 2*self.delta / np.prod(self.delta.shape)


class HingeLoss(Module):
    # Differentiable approximation of the 0-1 loss.
    # The perceptron update corresponds to a margin of 0, while the SVM loss
    # uses a margin of 1 (and regularization). See
    # https://stats.stackexchange.com/a/369480 for details. As an aside,
    # quadratic programming solvers converge much faster than SGD for
    # quadratic models such as SVM.

    def __init__(self, margin=1):
        self.margin = margin

    def forward(self, y_hat, y):
        # Note that y in [-1, 1]
        self.y_hat = y_hat
        self.y = y
        return np.maximum(0, self.margin - y_hat*y)

    def backward(self, grad=1):
        grad = self.y.copy()
        grad[self.y_hat*self.y > self.margin] = 0
        return grad


# Compositions of Modules
class Sequential(Module):
    def __init__(self, *modules, flatten=False):
        self.modules = modules

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, grad):
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def __repr__(self):
        if not self.modules:
            return super().__repr__()
        reprs = [repr(m).split('\n') for m in self.modules]
        reprs = '\n'.join(['\n'.join('  '+s for s in rows) for rows in reprs])
        return 'Sequential(\n' + reprs + '\n)'


class Add(Module):
    # Example of backprop with multiple inputs--just backprop on each input

    def __init__(self, *modules):
        self.modules = modules

    def forward(self, *xs):
        return sum(module.forward(x) for x, module in zip(xs, self.modules))

    def backward(self, grad):
        return [module.backward(grad) for module in self.modules]
