## Contents
This is my implementation of neural networks in numpy.

`layers.py` implements and explains these core layers:
- Linear, including weight initialization
- Binary logistic regression: LogSigmoid, BCELoss, BCEWithLogitsLoss
- Multinomial logistic regression: Softmax, LogSoftmax, NLLLoss, CrossEntropyLoss
- Activations: ReLU, Tanh, LeakyReLU
- Losses: Mean, MSELoss, HingeLoss
- Compositions: Sequential, Add

`conv2d.py` implements 2D convolution via im2col.

`optimizers.py` implements:
- SGD, including Nesterov momentum
- Adam, including AMSGrad

`neural_net.py` implements scikit-learn models:
- BinaryMLPClassifier
- MLPClassifier
- MLPRegressor
- Training loop examples

## Architecture
I represent each differentiable operation as a stateful `Module`. Each `Module` is responsible for computing relevant gradients and saving any intermediate state needed to do so. In contrast, PyTorch and Tensorflow have machinery to manage the compute graph and intermediate buffers, allowing non-trainable layers to be stateless.

`Module.backward()` takes the output gradient, computes any parameter gradients, and returns the input gradient. By [variable] gradient, I mean the gradient of the scalar loss with respect to [variable]. The loss layer takes output gradient 1 since the gradient of the loss with respect to itself is 1.

Operations can be composed using `Module`s like `Sequential` and `Add`. Their `backward()` methods handle the process of chaining `backward()` calls, so that we can compute gradients for all the trainable parameters in a model with a single `backward()` method.
