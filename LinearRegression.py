class my_LinearRegressor:
    
    def __init__(self, alpha, n_iters):
        
        self.alpha = alpha
        self.n_iters = n_iters
    
    def cost_function(self, X, Y):
        return np.sqrt(np.sum(np.square(self.predict(X) - Y))/self.m)
    
    def fit(self, X, Y):
        
        self.m , self.n = X.shape
        
        self.W = np.zeros(self.n)
        self.b = 0
        
        Jo = self.cost_function(X, Y)
        
        for i in range(self.n_iters):
            
            dW = 2/self.m *  np.dot(X.T, self.predict(X) - Y) 
            db = 2 * np.sum(Y - self.predict(X) ) / self.m
            
            self.W = self.W - self.alpha*dW
            self.b = self.b - self.alpha*db
            
            J = self.cost_function(X, Y)
            
    
    def predict(self, X):
        return np.dot(X, self.W) + self.b

