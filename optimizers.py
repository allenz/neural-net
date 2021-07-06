import numpy as np


class Optimizer:
    def zero_grad(self):
        for param, grad in self.parameters:
            grad[:] = 0


class SimpleSGD(Optimizer):
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        # Take a step down the loss gradient
        for param, grad in self.parameters:
            param -= self.lr * grad


class SGD(Optimizer):
    # Stochastic gradient descent with momentum.
    
    def __init__(self, parameters, lr, momentum=0, dampening=0,
                 nesterov=False):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov

        # stores the velocity of each parameter
        self.bufs = [None]*len(parameters)

    def step(self):
        for i, (param, grad) in enumerate(self.parameters):
            if self.momentum != 0:
                if self.bufs[i] is None:
                    self.bufs[i] = grad
                else:
                    self.bufs[i] *= self.momentum
                    self.bufs[i] += (1 - self.dampening) * grad

                if self.nesterov:
                    grad = grad + self.momentum * self.bufs[i]
                else:
                    grad = self.bufs[i]

            param -= self.lr * grad


class Adam(Optimizer):
    # SGD method that adapts step size for each parameter /  based on
    # estimates of first-order and second-order geometry.
    # - beta_1, beta_2: decay rates for first- and second-order moments
    # - eps: constant added to the denominator for numerical stability
    # - amsgrad: if True, compute the denominator based on the maximum
    #   second-order moment, which in theory improves convergence
    def __init__(self, parameters, lr, beta_1=0.9, beta2=0.999, eps=1e-7,
                 amsgrad=False, weight_decay=0):
        self.parameters = parameters
        self.lr = lr
        self.beta_1 = beta_1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad
        self.weight_decay = weight_decay

        self._step = 0

        # Exponential moving averages of gradients and squared gradients
        self.ema = [None]*len(parameters)
        self.ema2 = [None]*len(parameters)
        self.max_ema2 = [None]*len(parameters)

    def step(self):
        for i, (param, grad) in enumerate(self.parameters):
            if self.weight_decay != 0:
                grad *= 1 + self.weight_decay

            # initialize buffers
            if self.ema[i] is None:
                self.ema[i] = np.zeros_like(grad)
                self.ema2[i] = np.zeros_like(grad)
                if self.amsgrad:
                    self.max_ema2[i] = np.zeros_like(grad)

            # update moving averages
            self.ema[i] *= self.beta_1
            self.ema[i] += (1 - self.beta_1) * grad
            self.ema2[i] *= self.beta2
            self.ema2[i] += (1 - self.beta2) * grad**2

            denom = self.ema2[i]
            if amsgrad:
                self.max_ema2[i] = np.maximum(self.max_ema2[i], self.ema2[i])
                denom = self.max_ema2[i]
            
            bias_correction2 = 1 - self.beta2**self._step
            denom = (denom / bias_correction2).sqrt() + self.eps

            moment_1 = self.ema[i] / (1 - self.beta_1**self._step)
            param += self.lr * moment_1 / denom
            self._step += 1
