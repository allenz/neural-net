import numpy as np
import torch

from layers import (MSELoss, Sigmoid, LogSigmoid, BCELoss, BCEWithLogitsLoss,
                    Softmax, LogSoftmax, NLLLoss, CrossEntropyLoss, Linear)
from optimizers import SGD


def test_torch():
    # Check our outputs and gradients against PyTorch
    rng = np.random.default_rng(1)

    _test_torch(MSELoss, torch.nn.MSELoss)
    _test_torch(BCELoss, torch.nn.BCELoss)
    _test_torch(BCEWithLogitsLoss, torch.nn.BCEWithLogitsLoss)
    
    b, d = 3,5
    log_probs = scores = rng.standard_normal((b, d))
    y = rng.integers(0, d, (b,))
    _test_torch(NLLLoss, torch.nn.NLLLoss, log_probs, y)
    _test_torch(CrossEntropyLoss, torch.nn.CrossEntropyLoss, scores, y)

    scores = np.array([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]], dtype=np.single)
    _test_torch(Softmax, lambda: torch.nn.Softmax(dim=-1), scores)
    _test_torch(LogSoftmax, lambda: torch.nn.LogSoftmax(dim=-1), scores)


def _test_torch(our_fn, torch_fn, *args, test_grad=True):
    # Raises AssertionError if our results differ from PyTorch
    
    # Setup args, which defaults to (x, y) each of shape (2,1)
    if not args:
        x = np.array([[0.3], [0.5]], dtype=np.single)
        y = np.array([[0.2], [0.5]], dtype=np.single)
        args = (x,y)
    torch_args = [torch.tensor(a) for a in args]
    torch_args[0].requires_grad = True

    # Compare output
    f = our_fn()
    out = f.forward(*args)
    torch_out = torch_fn()(*torch_args)
    assert np.allclose(out, torch_out.detach().numpy())

    # Compare gradients
    if test_grad and not out.shape:
        grad = f.backward()
        torch_out.backward()
        assert np.allclose(grad, torch_args[0].grad)
    elif test_grad:
        # Reduce the output to a scalar by computing MSE against a vector of
        # all ones. Another option is to compute the mean:
        # reduction, torch_reduction, rargs = Mean, (lambda x: x.mean()), []
        reduction, torch_reduction = MSELoss, torch.nn.MSELoss()
        rargs = [np.ones_like(out)]
        torch_rargs = [torch.tensor(a) for a in rargs]
        r = reduction()
        scalar = r.forward(out, *rargs)
        grad = r.backward()
        grad = f.backward(grad)

        scalar_t = torch_reduction(torch_out, *torch_rargs)
        scalar_t.backward()
        grad2 = torch_args[0].grad
        assert (np.isclose(grad, grad2)).all(), str(grad) + '\n' + str(grad2)


def test_linear():
    # Train a Linear layer with MSELoss and SGD, and compare against PyTorch.

    # set up dataset and model
    x = np.array([[0.3, 0.2, 0.4]], dtype=np.single)
    y = np.array([[0.2, 0.5]], dtype=np.single)

    linear = Linear(3, 2)
    linear.w[:] = np.array([[-0.5, 0.1], [-0.3, -0.2], [0.8, 0.2]])
    linear.b[:] = 0
    loss = MSELoss()
    opt = SGD(linear.parameters(), lr=2, momentum=1, dampening=0.5)

    torch_linear = torch.nn.Linear(3, 2)
    with torch.no_grad():
        torch_linear._parameters['weight'][:] = torch.tensor(linear.w.T)
        torch_linear._parameters['bias'].zero_()
    torch_opt = torch.optim.SGD(torch_linear.parameters(), lr=2, momentum=1,
                                dampening=0.5)

    # check output
    y_hat = linear(x)
    out = loss(y_hat, y)

    tensor_x = torch.tensor(x, requires_grad=True)
    tensor_y_hat = torch_linear(tensor_x)
    torch_out = torch.nn.MSELoss()(tensor_y_hat, torch.tensor(y))

    assert np.allclose(out, torch_out.detach().numpy())

    # check gradients
    torch_out.backward()
    grad_in = linear.backward(loss.backward())
    assert np.allclose(grad_in, tensor_x.grad.numpy())

    assert np.allclose(linear.grad_w,
                       torch_linear._parameters['weight'].grad.T.numpy())
    assert np.allclose(linear.grad_b.ravel(),
                       torch_linear._parameters['bias'].grad.numpy())

    # check optimizer
    # take two steps to test momentum
    opt.step()
    opt.step()
    torch_opt.step()
    torch_opt.step()

    assert np.allclose(linear.w,
                       torch_linear._parameters['weight'].T.detach().numpy())
    assert np.allclose(linear.b.ravel(),
                       torch_linear._parameters['bias'].detach().numpy())


def test_consistency():
    # Check that Sigmoid + BCELoss = BCEWithLogitsLoss

    logits = np.array([[0]], dtype=np.single)
    sigmoid = Sigmoid()
    y_hat = sigmoid.forward(logits)
    y = np.array([[0.6]], dtype=np.single)

    f = BCELoss()
    loss = f.forward(y_hat, y)
    grad = sigmoid.backward(f.backward())

    f2 = BCEWithLogitsLoss()
    loss2 = f2.forward(logits, y)
    grad2 = f2.backward()

    assert np.isclose(loss, loss2)
    assert np.isclose(grad, grad2)
