def gradient_descent(θ1, θ2, alpha, n_iter):
    for _ in range(n_iter):
        for x, y in zip(X, Y):
            # calculate gradient with respect to weights
            θ1_grad, θ2_grad = gradient_func(x, y, θ1, θ2)
            
            # update parameters using grad
            θ1 = θ1 - alpha * θ1_grad
            θ2 = θ2 - alpha * θ2_grad
    
    # we now have optimized values for parameters
    return θ1, θ2

def gradient_func(x, y, θ1, θ2):
    error = get_pred(θ1, θ2, x) - y
    
    θ1_grad = (2 * error) * x
    θ2_grad = (2 * error)
    
    return θ1_grad, θ2_grad

def get_pred(x, θ1, θ2):
    y_pred = θ1 * x + θ2
    return y_pred
            
if __name__ == "__main__":
    X = list(range(10))
    Y = [2*x for x in X] # y = 2x
    
    # Define parameters assuming y = θ1x + θ2
    import numpy as np
    θ1 = np.random.normal(0, 1)
    θ2 = np.random.normal(0, 1)
    alpha = 0.0001
    n_iter = 1000
    
    θ1_final, θ2_final = gradient_descent(θ1, θ2, alpha, n_iter)
    
    print(θ1_final, θ2_final)