def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        
    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        
        for _ in range(self.iterations):
            linear_model = np.dot(X, self.W) + self.b
            y_predicted = sigmoid(linear_model)
            
            # Gradient Descent
            dw = (1 / self.m) * np.dot(X.T, (y_predicted - y))
            db = (1 / self.m) * np.sum(y_predicted - y)
            
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
    def predict(self, X):
        linear_model = np.dot(X, self.W) + self.b
        y_predicted = sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]