# 2D convolutions implemented in numpy.

import numpy as np

from layers import Module


def output_size(H_in, W_in, kernel_size, stride, padding):
    # Returns the output shape given input shape, stride, padding, kernel size
    H_out = (H_in + 2*padding - kernel_size) // stride + 1
    W_out = (W_in + 2*padding - kernel_size) // stride + 1
    return H_out, W_out


class Conv2D(Module):
    # Applies a 2D convolution.
    # Takes input (B, H_in, W_in, C_in) and returns (B, H_out, W_out, C_out).
    # A grouped convolution divides the input channels into groups and
    # convolves each group separately, reducing model size and FLOPs.
    # In a depthwise convolution, groups = C_in.
    # Parameters: weight w (k*k*C_in/groups, C_out) and bias b (C_out,).
    # w is initialized with He uniform and b is initialized to 0.

    def __init__(self, C_in, C_out, kernel_size,
                 stride=1, padding=0, groups=1):
        
        assert C_in % groups == 0 and C_out % groups == 0

        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # Initialize weight and bias.
        # Field size is the number of inputs for each convolutional unit.
        # Scaling by 1/sqrt(n) gives us approximately unit variance.
        field_size = self.kernel_size**2 * C_in // self.groups
        scale = 1/np.sqrt(field_size)
        unif = np.random.rand(field_size, self.C_out).astype(np.float32)
        self.w = 2 * scale * unif - scale
        self.b = np.zeros((1, self.C_out), dtype=np.float32)

    def im2col(self, im, H_in, W_in, H_out, W_out):
        # Organizes an image (H_in, W_in, C_in) into a matrix where each column
        # represents a kernel's receptive field. This helps vectorize spatial
        # operations at the cost of higher memory usage. A faster
        # implementation would use np.lib.stride_tricks.as_strided to create
        # a view.
        # Returns (kernel_size * kernel_size * C_in, H_out * W_out).

        k = self.kernel_size
        stride = self.stride
        pad = self.padding

        im = np.pad(im, [(pad, pad), (pad, pad), (0, 0)])
        col = np.zeros((k*k*self.C_in, H_out*W_out), dtype=im.dtype)
        for r in range(H_out):
            for c in range(W_out):
                region = im[stride*r : stride*r + k, stride*c : stride*c + k]
                col[:, W_out*r + c] = region.ravel()
        return col

    def col2im(self, col, H_in, W_in, H_out, W_out):
        # Given a matrix of receptive fields, returns the original image.
        # Helps vectorize the backward pass. Overlapping regions are summed.

        k = self.kernel_size
        stride = self.stride
        pad = self.padding

        im = np.zeros((H_in + 2*pad, W_in + 2*pad, self.C_in), dtype=col.dtype)
        for r in range(H_out):
            for c in range(W_out):
                region = col[:, W_out*r + c].reshape((k, k, self.C_in))
                im[stride*r : stride*r + k,
                   stride*c : stride*c + k, :] += region
        return im[pad:H_in+pad, pad:W_in+pad, :]

    def forward(self, x):
        B, H_in, W_in, C_in = x.shape
        H_out, W_out = output_size(H_in, W_in, self.kernel_size,
                                   self.stride, self.padding)
        col_size = self.kernel_size**2 * C_in

        # Save for backward pass
        self.x = x
        self.H_in, self.W_in = H_in, W_in

        out = np.zeros((B, H_out * W_out, self.C_out), dtype=x.dtype)
        for ix in range(B):
            col = self.im2col(x[ix], H_in, W_in, C_in, H_out, W_out)
            for g in range(self.groups):
                # select the patches for the current group
                col_g = col[   g    * col_size // self.groups :
                            (g + 1) * col_size // self.groups]

                # select parameters for the current group
                C_start = g * self.C_out // self.groups
                C_end = (g + 1) * self.C_out // self.groups
                w = self.w[:, C_start:C_end]
                b = self.b[:, C_start:C_end]

                out[ix, :, C_start:C_end] = col_g.T @ w + b
        
        return out.reshape(B, H_out, W_out, self.C_out)

    def backward(self, grad):
        B, H_out, W_out, C_out = grad.shape
        H_in, W_in = self.H_in, self.W_in
        col_size = self.kernel_size**2 * self.C_in

        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)
        grad_in = np.zeros_like(self.x)

        for ix in range(B):
            col = self.im2col(self.x[ix], H_in, W_in, H_out, W_out)
            grad_col = np.zeros_like(col)
            for g in range(self.groups):
                # slice the input column for the current group
                col_g = col[   g    * col_size // self.groups :
                            (g + 1) * col_size // self.groups]

                C_start = g * C_out // self.groups
                C_end = (g + 1) * C_out // self.groups
                grad_g = grad[ix, :, :, C_start:C_end]
                w_g = self.w[:, C_start:C_end]
                
                # compute gradients
                self.grad_w[:, C_start:C_end] += col_g @ grad_g / B
                self.grad_b[:, C_start:C_end] += grad_g.mean(axis=0)
                grad_col[curr_g:next_g, :] = w_g @ grad_g.T
            
            grad_in[ix] = self.col2im(grad_col, H_in, W_in, H_out, W_out)
        return grad_in
