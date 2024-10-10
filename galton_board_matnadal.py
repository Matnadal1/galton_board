import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy as scipy
from scipy.interpolate import make_interp_spline

# Step 1: Simulate the Galton Board
def simulate_galton_board(n, N):
    bins = np.zeros(n+1)
    for _ in range(N):
        position = 0
        for i in range(n):
            if np.random.rand() > 0.5:
                position += 1
        bins[position] += 1
    return bins

# Step 2: Compute Binomial Probabilities
def binomial_probabilities(n):
    probabilities = np.array([scipy.special.comb(n, k) * (0.5**n) for k in range(n+1)])
    return probabilities

# Step 3: Compare with Normal Distribution
def normal_distribution(n, N):
    mu = n / 2
    sigma = np.sqrt(n / 4)
    x = np.arange(0, n+1)
    normal_prob = N * norm.pdf(x, mu, sigma)
    return x, normal_prob

# Step 4: Smooth curves using interpolation
def smooth_curve(x, y):
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# Step 5: Plot Results and Error
def plot_results(n, N, bins, binomial_probs, normal_x, normal_prob):
    experimental_probs = bins / np.sum(bins)
    
    x_binomial_smooth, binomial_smooth = smooth_curve(np.arange(0, n+1), binomial_probs * N)
    x_normal_smooth, normal_smooth = smooth_curve(normal_x, normal_prob)
    
    plt.figure(figsize=(10, 6))
    x = np.arange(0, n+1)
    
    for i in range(len(bins)):
        plt.bar(i, bins[i], width=0.3, alpha=0.6, color='blue', label='Experimental' if i == 0 else "")

    plt.plot(x_binomial_smooth, binomial_smooth, label='Theoretical Binomial', color='green', linestyle='-', marker=None)
    plt.plot(x_normal_smooth, normal_smooth, label='Normal Distribution', color='red', linestyle='--')

    plt.xlabel('Position')
    plt.ylabel('Number of Balls')
    plt.title(f'Galton Board Simulation (n={n}, N={N})')
    plt.legend()
    plt.show()

# Step 6: Compute Mean Squared Error
def mean_squared_error(experimental, theoretical):
    return np.mean((experimental - theoretical)**2)

n = 5  # Number of levels in the Galton board
N = 20  # Number of balls

bins = simulate_galton_board(n, N)
binomial_probs = binomial_probabilities(n)
normal_x, normal_prob = normal_distribution(n, N)

plot_results(n, N, bins, binomial_probs, normal_x, normal_prob)

experimental_probs = bins / N
mse = mean_squared_error(experimental_probs, binomial_probs)
print(f'Mean Squared Error for n = {n} and N = {N} : {mse}')
