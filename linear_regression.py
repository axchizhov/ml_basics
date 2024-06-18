# %%
import numpy as np

np.random.seed(42)


class LinearRegression:
    def __init__(self):
        self.weights = None
        self.learning_rate = 0.001

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # self.weights = np.random.randn(n_features, 1)
        self.weights = np.zeros((n_features, 1))

        epochs = 10

        for _ in range(epochs):
            y_pred = self.predict(X)

            loss = (y - y_pred) ** 2
            loss = sum(loss) / n_samples
            print(loss)

            grad = -2 * X.T @ (y - y_pred)
            self.weights -= self.learning_rate * grad

    def predict(self, X):
        y_pred = X @ self.weights
        return y_pred


# %%

X = np.random.randn(50, 10)
bias_column = np.ones((X.shape[0], 1))
X = np.hstack((X, bias_column))

y = np.random.randn(50, 1)

model = LinearRegression()

model.fit(X, y)
