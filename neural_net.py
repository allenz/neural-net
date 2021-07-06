from numpy import np

from layers import (Module, Linear, Dense, ReLU, MSELoss, Sigmoid, LogSigmoid, 
                    BCEWithLogitsLoss, CrossEntropyLoss, Sequential, MLP)


def train_test_split(X, y, frac):
    X, y = np.array(X), np.array(y)
    n_train = int((1-frac)*len(y))
    n_test = len(y) - n_train
    ix = np.random.permutation(len(y))
    test_i, train_i = ix[n_test:], ix[:n_test]
    return X[test_i], y[test_i], X[train_i], y[train_i]


class Estimator(Module):
    # Adapts model/loss modules implementing forward() and backward() to the
    # scikit-learn fit() and predict() api.
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss

    def forward(self, x, y):
        y_hat = self.model.forward(x)
        return self.loss.forward(y_hat, y)

    def backward(self, grad, lr):
        grad = self.loss.backward(grad, lr)
        return self.model.backward(grad, lr)

    def train(self, X, y, bs, lr): # one epoch
        ix = np.random.permutation(len(y))
        X, y = X[ix], y[ix]
        loss = 0
        for i in range(0, len(y), bs):
            loss += self.forward(X[i:i+bs], y[i:i+bs])
            self.backward(grad=1, lr=lr)
        return loss/(i+1)

    def fit(self, X, y, epochs=10, bs=4, lr=1e-4):
        X, y = np.array(X), np.array(y)
        for ep in range(epochs):
            self.train(X, y, bs, lr)

    def fit_track(self, X, y, epochs=100, bs=1, lr=None, frac=0.1):
        # Convenience function to fit the dataset and log train/test losses.
        # Takes a full dataset (X, y) and makes a train-test split
        # automatically.
        lr = lr or 0.1*len(y)/(X.shape[1]**2)
        train_X, train_y, test_X, test_y = train_test_split(X, y, frac)
        for ep in range(epochs):
            loss = self.train(train_X, train_y, bs, lr)
            if epochs < 10 or ep % (epochs//10) == 0:
                test_loss = self.forward(test_X, test_y)
                print(f'epoch {ep:2}: train {loss:.2e}, test {test_loss:.2e}')


def MLP(d_in, d_out, hidden_dims, activation):
    dims = [d_in] + hidden_dims
    hiddens = [Dense(a, b, activation) for a, b in zip(dims, dims[1:])]
    return Sequential(*hiddens, Linear(dims[-1], d_out))


class MLPRegressor(Estimator):
    # This is multinomial logistic regression if there are no hidden layers
    def __init__(self, d_in, d_out=1, hidden_dims=[], activation=ReLU):
        self.model = MLP(d_in, d_out, hidden_dims, activation)
        self.loss = MSELoss()


class BinaryMLPClassifier(Estimator):
    # This is binary logistic classification if there are no hidden layers
    def __init__(self, d_in, hidden_dims=[], activation=ReLU):
        # This is more numerically stable than using Sigmoid and BCELoss
        self.model = MLP(d_in, 1, hidden_dims, activation)
        self.loss = BCEWithLogitsLoss()
    
    def predict_log_proba(self, X):
        return LogSigmoid()(self.model.forward(X))

    def predict(self, X, db=0.5):
        return self.predict_log_proba(X) >= np.log(db)


class MLPClassifier(Estimator):
    # This is multinomial logistic classification if there are no hidden layers
    def __init__(self, d_in, d_out, hidden_dims=[], activation=ReLU):
        self.model = MLP(d_in, 1, hidden_dims, activation)
        self.loss = CrossEntropyLoss()


def regression_example():
    n, d = 400, 100
    target_fn = Linear(d_in=d, d_out=1)

    # make some data
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, d))
    y = target_fn(X)

    # fit regressor
    X, y = regression_data()
    clf = MLPRegressor(d_in=X.shape[1])
    clf.fit_track(X, y)


def classification_example():
    n = 5
    target_fn = Dense(2, 1, Sigmoid)

    # make some data
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, 2))
    y = target_fn(X) > 0.5
    print('true', y)
    
    # fit classifier
    clf = BinaryMLPClassifier(d_in=2)
    clf.fit_track(X, y, bs=1, lr=0.1)
    print('pred', clf.predict(X))


if __name__ == "__main__":
    classification_example()
