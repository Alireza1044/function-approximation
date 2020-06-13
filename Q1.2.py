import numpy as np
import matplotlib.pyplot as plt


class RBFLayer:
    def __init__(self, bases):
        self.bases = bases
        self.centers = np.ndarray((bases, 1))
        self.stds = np.ndarray((bases, 1))
        self.A = np.ndarray((bases, 1))

    def kmeans(self, X, k):
        clusters = np.random.choice(np.squeeze(X), size=k)
        prevClusters = clusters.copy()
        converged = False

        while not converged:
            distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))

            closestCluster = np.argmin(distances, axis=1)

            for i in range(k):
                pointsForCluster = X[closestCluster == i]
                if len(pointsForCluster) > 0:
                    clusters[i] = np.mean(pointsForCluster, axis=0)

            converged = np.linalg.norm(clusters - prevClusters) < 1e-6
            prevClusters = clusters.copy()

        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
        closestCluster = np.argmin(distances, axis=1)

        clustersWithNoPoints = []
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) < 2:
                clustersWithNoPoints.append(i)
                continue
        if len(clustersWithNoPoints) > 0:
            pointsToAverage = []
            for i in range(k):
                if i not in clustersWithNoPoints:
                    pointsToAverage.append(X[closestCluster == i])

        return np.reshape(clusters, (self.bases, 1))


class OutputLayer:
    def __init__(self, bases):
        self.W = np.zeros((bases, 1))
        self.A = 0


class RBF:
    def __init__(self, bases, sigma):
        self.bases = bases
        self.sigma = sigma
        self.hidden_layer = RBFLayer(bases)
        self.output_layer = OutputLayer(bases)

    def gaussian(self, x, c):
        return np.exp((-self.sigma * (x - c.T) ** 2))

    def linear(self):
        return self.hidden_layer.A.dot(self.output_layer.W)

    def forward(self, x):
        self.hidden_layer.A = self.gaussian(x, self.hidden_layer.centers)
        self.output_layer.A = self.linear()[0][0]

    def fit(self, input, output):
        self.hidden_layer.centers = self.hidden_layer.kmeans(input, self.bases)
        self.forward(input)
        self.hidden_layer.A = np.linalg.pinv(self.hidden_layer.A)
        self.output_layer.W = np.dot(self.hidden_layer.A, output)

    def predict(self, input):
        result = []
        for i in range(len(input)):
            self.forward(input[i])
            result.append(self.output_layer.A)
        return result


if __name__ == '__main__':
    train_data_count = 11000
    test_data_count = 300

    x_train = np.random.uniform(low=-100, high=100, size=train_data_count)
    x_train = np.sort(x_train, axis=0)
    noise = np.random.uniform(-10, 10, train_data_count)
    y_train = np.power(x_train, 2)  # + noise
    x_train = np.reshape(x_train, (train_data_count, 1))
    y_train = np.reshape(y_train, (train_data_count, 1))

    nn = RBF(bases=65, sigma=0.05)
    nn.fit(x_train, y_train)

    output = nn.predict(x_train)

    plt.title("RBF NN Train Results")
    plt.plot(x_train, output, label='Predicted')
    plt.plot(x_train, np.power(x_train, 2), label='Excpected')
    plt.legend()
    plt.show()

    test = np.linspace(-3, 3, test_data_count)
    test = np.reshape(test, (300, 1))
    output = nn.predict(test)

    plt.title("RBF NN [-3,3]")
    plt.plot(test, output, label='Predicted')
    plt.plot(test, np.power(test, 2), label='Excpected')
    plt.legend()
    plt.show()
