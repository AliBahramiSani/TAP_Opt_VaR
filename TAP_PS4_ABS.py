#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import time
from scipy.stats import norm
from scipy.stats import gaussian_kde

#%%
# 1.1 Polynomial
def f(x):
    return -x**6 + 2*x**3 + 3*x**2

def df(x):
    return -6*x**5 + 6*x**2 + 6*x

def ddf(x):
    return -30*x**4 + 12*x + 6

# Newton-Raphson method
def newton_raphson(x0, tol=1e-6, max_iter=300):
    x = x0
    for i in range(max_iter):
        dfx = ddf(x)
        if abs(dfx) < 1e-10:  # Avoid division by zero
            return None, i
        x_new = x - df(x) / dfx
        if abs(x_new - x) < tol:
            return x_new, i
        x = x_new
    return None, max_iter  # Return None if it does not converge

# %%
# Define a grid of initial guesses
x_vals = np.linspace(-3, 3, 100)
solutions = []
iterations = []

for x0 in x_vals:
    root, iters = newton_raphson(x0)
    solutions.append(root if root is not None else np.nan)
    iterations.append(iters)

# # Find the global maximum with built in functions
# def neg_f(x):
#     return -f(x)

# # Use scipy's minimize_scalar to find the global maximum
# result = minimize_scalar(neg_f, bounds=(-3, 3), method='bounded')
# global_maximizer = result.x
# global_maximum = f(global_maximizer)

# global_maximizer, global_maximum
# %%
# Plot the obtained solutions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_vals, solutions, c='b', label="Converged Solution")
# plt.axhline(global_maximizer, color='r', linestyle='--', label="Global Maximum with built in functions")
plt.axhline(y = max(solutions), color='r', linestyle='-', label="Global Maximum with Newton-Raphson loop")
plt.xlabel("Initial Guess")
plt.ylabel("Solution")
plt.title("Solution vs Initial Guess")
plt.legend()

# Plot the number of iterations
plt.subplot(1, 2, 2)
plt.scatter(x_vals, iterations, c='g', label="Iterations")
plt.xlabel("Initial Guess")
plt.ylabel("Iterations")
plt.title("Iterations vs Initial Guess")
plt.legend()

plt.tight_layout()
plt.savefig("Plots/NR_X0.pdf", format="pdf")
plt.show()
# %%
# Compute the share of initial guesses that led to the highest solution found
global_tolerance = 1e-1  # Allow small numerical error
max_solution = np.nanmax(solutions)  # Get the highest solution found
success_count_max = sum(1 for sol in solutions if abs(sol - max_solution) < global_tolerance)
total_count = len(solutions)
success_share_max = success_count_max / total_count  # Compute share

print(success_share_max*100,"%")

# %%
# 1.2 Sigmoid function
# Define the modified sigmoid function and its derivative
def sigmoid_eq(x):
    return 1 / (1 + np.exp(-x)) - 0.5

def d_sigmoid_eq(x):
    return np.exp(-x) / (1 + np.exp(-x))**2  # Derivative of sigmoid function


# %%
# Run Newton-Raphson for a range of initial guesses
x_vals_sigmoid = np.linspace(-3, 3, 150)
solutions_sigmoid = []
iterations_sigmoid = []
avg_update_steps = []

for x0 in x_vals_sigmoid:
    x = x0
    update_steps = []
    for i in range(150):  # Max 150 iterations
        dfx = d_sigmoid_eq(x)
        if abs(dfx) < 1e-10:  # Avoid division by zero
            solutions_sigmoid.append(None)
            iterations_sigmoid.append(i)
            avg_update_steps.append(np.nan)
            break
        x_new = x - sigmoid_eq(x) / dfx
        update_steps.append(abs(x_new - x))
        if abs(x_new - x) < 1e-6:
            solutions_sigmoid.append(x_new)
            iterations_sigmoid.append(i)
            avg_update_steps.append(np.mean(update_steps))
            break
        x = x_new
    else:
        solutions_sigmoid.append(None)  # No convergence
        iterations_sigmoid.append(100)
        avg_update_steps.append(np.nan)

# Process solutions to identify divergence
solutions_clipped = []
colors = []

for sol in solutions_sigmoid:
    if sol is None or np.isnan(sol):  # If the method failed
        solutions_clipped.append(3 if np.random.rand() > 0.5 else -3)  # Assign ±3
        colors.append('r')  # Red for divergence
    else:
        solutions_clipped.append(sol)
        colors.append('b')  # Blue for convergence
#%%
# Separate converged and diverged points
x_converged = [x_vals_sigmoid[i] for i in range(len(solutions_clipped)) if colors[i] == 'b']
y_converged = [solutions_clipped[i] for i in range(len(solutions_clipped)) if colors[i] == 'b']

x_diverged = [x_vals_sigmoid[i] for i in range(len(solutions_clipped)) if colors[i] == 'r']
y_diverged = [solutions_clipped[i] for i in range(len(solutions_clipped)) if colors[i] == 'r']

# Plot separately
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Solutions plot with separated dots
axs[0].scatter(x_converged, y_converged, c='b', label="Converged")
axs[0].scatter(x_diverged, y_diverged, c='r', label="Diverged (Capped at ±3)")
axs[0].axhline(y=0, color='g', linestyle='--', label="True Solution (x=0)")
# axs[0].axhline(y=3, color='r', linestyle=':', label="Diverged Above +3")
# axs[0].axhline(y=-3, color='r', linestyle=':', label="Diverged Below -3")
axs[0].set_xlabel("Initial Guess")
axs[0].set_ylabel("Iteration Solutions")
axs[0].set_title("Iteration Solutions vs Initial Guess")
axs[0].legend()

# Average update steps plot
axs[1].scatter(x_vals_sigmoid, avg_update_steps, c='g', label="Avg. Update Step")
axs[1].set_xlabel("Initial Guess")
axs[1].set_ylabel("Avg. Update Step")
axs[1].set_title("Average Update Step vs Initial Guess")
axs[1].legend(loc = 'upper right')

plt.tight_layout()
plt.savefig("Plots/NR_Sigmoid.pdf", format="pdf")
plt.show()


# %%
# Bisection method
def bisection(a, b, tol=1e-6, max_iter=100):
    if df(a) * df(b) >= 0:
        raise ValueError("The function must have opposite signs at a and b.")
    iterations = 0
    while (b - a) / 2 > tol and iterations < max_iter:
        c = (a + b) / 2
        if df(c) == 0:
            break
        if df(a) * df(c) < 0:
            b = c
        else:
            a = c
        iterations += 1
    return (a + b) / 2, iterations

# Define a grid of intervals to search for roots
intervals = [(-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3)]
roots = []
iterations = []

# Apply the bisection method to each interval
for a, b in intervals:
    if df(a) * df(b) < 0:
        root, iters = bisection(a, b)
        roots.append(root)
        iterations.append(iters)

# Print results
print("Roots found:", roots)
print("Iterations required:", iterations)

# Plot the function and the roots
x_vals = np.linspace(-3, 3, 1000)
y_vals = df(x_vals)
plt.plot(x_vals, y_vals, label="f(x)")
plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
plt.scatter(roots, [0] * len(roots), color="red", label=f"Root: {[round(root, 2) for root in roots]}")
plt.xlabel("x")
plt.ylabel("df(x)")
# plt.title("Bisection Method: Roots of f(x)")
plt.legend()
# plt.grid()
plt.savefig("Plots/Bisection.pdf", format="pdf")
plt.show()
# %%
# Integral Section
np.random.seed(137)
# Define the function
def f(x):
    return -x**6 + 2*x**3 + 3*x**2

# Exact value of the integral (computed analytically)
exact_integral = 4.12345  # Replace with the exact value

# Trapezoidal Rule
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    integral = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        integral += f(a + i * h)
    integral *= h
    return integral

# Simpson’s 1/3 Rule
def simpsons_rule(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson’s Rule.")
    h = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 1:
            integral += 4 * f(x)
        else:
            integral += 2 * f(x)
    integral *= h / 3
    return integral

# Monte Carlo Integration
def monte_carlo_integration(f, a, b, N):
    x_samples = np.random.uniform(a, b, N)
    f_values = f(x_samples)
    integral = (b - a) * np.mean(f_values)
    return integral

# Integration limits
a = -1
b = 1.57474

# Values of N to test
N_values = [6, 10, 100, 500, 1000, 5000, 10000]

# Store results
results = {"Trapezoidal": [], "Simpson": [], "Monte Carlo": []}
errors = {"Trapezoidal": [], "Simpson": [], "Monte Carlo": []}
times = {"Trapezoidal": [], "Simpson": [], "Monte Carlo": []}

# Compute integrals and errors
for N in N_values:
    # Trapezoidal Rule
    start = time.time()
    trap = trapezoidal_rule(f, a, b, N)
    end = time.time()
    results["Trapezoidal"].append(trap)
    errors["Trapezoidal"].append(abs(trap - exact_integral))
    times["Trapezoidal"].append(end - start)

    # Simpson’s Rule
    start = time.time()
    simp = simpsons_rule(f, a, b, N)
    end = time.time()
    results["Simpson"].append(simp)
    errors["Simpson"].append(abs(simp - exact_integral))
    times["Simpson"].append(end - start)

    # Monte Carlo Integration
    start = time.time()
    mc = monte_carlo_integration(f, a, b, N)
    end = time.time()
    results["Monte Carlo"].append(mc)
    errors["Monte Carlo"].append(abs(mc - exact_integral))
    times["Monte Carlo"].append(end - start)

# Print results
print("Results:")
for method in results:
    print(f"{method}: {results[method]}")

print("\nErrors:")
for method in errors:
    print(f"{method}: {errors[method]}")

print("\nTimes:")
for method in times:
    print(f"{method}: {times[method]}")
# %%
# Plot absolute errors
plt.figure(figsize=(12, 6))
plt.semilogx(N_values, errors["Trapezoidal"], 'o-', label="Trapezoidal Rule")
plt.semilogx(N_values, errors["Simpson"], 's-', label="Simpson’s Rule")
plt.semilogx(N_values, errors["Monte Carlo"], 'd-', label="Monte Carlo")
# plt.plot(N_values, errors["Trapezoidal"], 'o-', label="Trapezoidal Rule")
# plt.plot(N_values, errors["Simpson"], 's-', label="Simpson’s Rule")
# plt.plot(N_values, errors["Monte Carlo"], 'd-', label="Monte Carlo")
plt.xlabel("N (log scale)")
plt.ylabel("Absolute Error")
# plt.title("Absolute Errors vs N")
plt.legend()
plt.grid()
plt.savefig("Plots/integrals.pdf", format="pdf")
plt.show()
# %%
# Portfolio selection and Value at Risk
# Alice's Portfolios
# Given data
r = np.array([0.12, 0.14, 0.16])  # Expected returns
Sigma = np.array([
    [0.04, 0.01, 0.02],
    [0.01, 0.09, 0.03],
    [0.02, 0.03, 0.16]
])  # Covariance matrix

# Vector of ones
ones = np.ones(len(r))

# Inverse of the covariance matrix
inv_cov = np.linalg.inv(Sigma)

# Compute A, B, C, D for the MVF
A = r.T @ inv_cov @ ones
B = r.T @ inv_cov @ r
C = ones.T @ inv_cov @ ones
D = B * C - A ** 2

# MVF without risk-free asset
def sigma_wo_riskfree(mu):
    return np.sqrt(1 / C + (C / D) * (mu - A / C) ** 2)

# MVF with risk-free asset
# r_0 = 0.05  # Assume a risk-free rate of 5%
# H = (r - r_0 * ones).T @ inv_cov @ (r - r_0 * ones)

# MVF without risk-free asset
mu_values = np.linspace(0.10, 0.20, 100)  # Target returns for Alice
sigma_values = sigma_wo_riskfree(mu_values)

# Tangency portfolio
# mu_hat = (B - A * r_0) / (A - C * r_0)
# sigma_hat = np.sqrt(H / (A - C * r_0) ** 2)
# %%
# Plot efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(sigma_values, mu_values, 'b-', label="Alice's Efficient Frontier")
plt.xlabel("Volatility")
plt.ylabel("Expected Return")
plt.legend()
plt.show()
# %%
# Portfolio weights for Alice's target returns
def portfolio_weights(mu_target, r, inv_cov, A, B, C, D):
    term1 = (B - A * mu_target) * (inv_cov @ ones)
    term2 = (C * mu_target - A) * (inv_cov @ r)
    w = (term1 + term2) / D
    return w

# Compute portfolios for Alice
mu_values = np.linspace(0.10, 0.20, 100)  # Target returns for Alice
weights = [portfolio_weights(mu, r, inv_cov, A, B, C, D) for mu in mu_values]
volatilities = [np.sqrt(w.T @ Sigma @ w) for w in weights]

# Check if portfolio weights sum to 1
# assert all(np.isclose(np.sum(w), 1) for w in weights), "Some portfolio weights do not sum to 1"
# Check if portfolio weights sum to 1
is_close_results = [np.isclose(np.sum(w), 1) for w in weights]
print(is_close_results)  # Print the results to debug
# Assert that all portfolio weights sum to 1
assert all(is_close_results), "Some portfolio weights do not sum to 1"
# [print(f"Portfolio weights sum to: {np.sum(w):.4f}") for w in weights]
# %%
# Bob's Monte Carlo simulation
M = 10000  # Number of simulations
np.random.seed(137)  # For reproducibility
returns_simulated = np.random.multivariate_normal(r, Sigma, M)

# Compute realized returns and volatilities for Alice's portfolios
realized_returns = np.mean(np.array([returns_simulated @ w for w in weights]), axis=1)
realized_volatilities = np.array([np.std(returns_simulated @ w) for w in weights])

# Confidence intervals for expected returns
CI_mu = np.array([
    realized_returns - 1.96 * realized_volatilities / np.sqrt(M),
    realized_returns + 1.96 * realized_volatilities / np.sqrt(M)
])
#%%
# Compute the Upper Bound
# Given parameters
lambda_ = 0.70
V_bar = 0.075
mu_min = 0.10

# Quantile for lambda = 0.70
z_lambda = norm.ppf(lambda_)  # z_lambda = -0.5244

# Upper bound on volatility
sigma_max = (V_bar + mu_min) / abs(z_lambda)
print(f"Upper bound on volatility: {sigma_max:.4f}")
# VaR constraint
lambda_ = 0.70  # Confidence level
V_bar = 0.075  # Maximum loss
# To specify which
VaR = np.array([-np.percentile(returns_simulated @ w, 100 * (1 - lambda_)) for w in weights])
VaR_compliant = VaR <= V_bar

# %%
# Plot efficient frontier with confidence intervals and VaR compliance
plt.figure(figsize=(10, 6))
plt.plot(volatilities, mu_values, 'b-', label="Efficient Frontier (No Risk-Free)")
plt.fill_between(volatilities, CI_mu[0], CI_mu[1], color='gray', alpha=0.3, label="95% CI")

# Highlight VaR-compliant portfolios
plt.scatter(np.array(realized_volatilities)[VaR_compliant], np.array(realized_returns)[VaR_compliant], color='green', label="VaR-Compliant")
plt.scatter(np.array(realized_volatilities)[~VaR_compliant], np.array(realized_returns)[~VaR_compliant], color='red', label="VaR-Violating")

plt.xlabel("Volatility")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier with Confidence Intervals and VaR Compliance")
plt.legend()
plt.grid()
plt.show()


# %%
# Comparing Theoretical and Empirical VaRs

## Theoretical VaR
# Quantile for lambda = 0.70
z_lambda = norm.ppf(1 - lambda_)  # z_lambda = -0.5244

# Upper bound on volatility
sigma_max = (V_bar + mu_min) / abs(z_lambda)
print(f"Upper bound on volatility: {sigma_max:.4f}")

# Alice's portfolios
# mu_values = np.linspace(0.10, 0.20, 100)  # Target returns
volatilities = [np.sqrt(w.T @ Sigma @ w) for w in weights]  # Portfolio volatilities

# Theoretical VaR compliance
theoretical_VaR = np.array([-(mu + z_lambda * sigma) for mu, sigma in zip(mu_values, volatilities)])
theoretical_VaR_compliant = theoretical_VaR <= V_bar
# %%

## Empirical VaR
# Compute empirical VaR for each portfolio
empirical_VaR = np.array([-np.percentile(returns_simulated @ w, 100 * (1 - lambda_)) for w in weights])

# Check empirical VaR compliance
empirical_VaR_compliant = empirical_VaR <= V_bar

# Compare theoretical and empirical VaR compliance
print(f"Theoretical VaR-compliant portfolios: {np.sum(theoretical_VaR_compliant)}")
print(f"Empirical VaR-compliant portfolios: {np.sum(empirical_VaR_compliant)}")



# %%
# Plot efficient frontier with confidence intervals and VaR compliance
plt.figure(figsize=(12, 6))
plt.plot(volatilities, mu_values, 'b-', label="Efficient Frontier")
plt.fill_between(volatilities, CI_mu[0], CI_mu[1], color='gray', alpha=0.3, label="95% CI")
# Highlight empirically VaR-compliant and VaR-violating portfolios
plt.scatter(np.array(volatilities)[empirical_VaR_compliant], np.array(mu_values)[empirical_VaR_compliant], color='green', label="VaR-Compliant (Empirical)")
plt.scatter(np.array(volatilities)[~empirical_VaR_compliant], np.array(mu_values)[~empirical_VaR_compliant], color='red', label="VaR-Violating (Empirical)")
plt.xlabel("Volatility")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier with Confidence Intervals and Empirical VaR Compliance")
plt.legend()
plt.grid()
plt.show()
# %%
# Compare theoretical and empirical VaR compliance
discrepancy = theoretical_VaR_compliant != empirical_VaR_compliant
print(f"Number of discrepancies: {np.sum(discrepancy)}")
print(f"Discrepant portfolios: {np.where(discrepancy)[0]}")
# %%
plt.figure(figsize=(12, 6))
plt.plot(volatilities, mu_values, 'b-', label="Efficient Frontier ")
plt.fill_between(volatilities, CI_mu[0], CI_mu[1], color='gray', alpha=0.3, label="95% CI")
# Highlight empirically VaR-compliant and VaR-violating portfolios
plt.scatter(np.array(volatilities)[empirical_VaR_compliant], np.array(mu_values)[empirical_VaR_compliant], color='green', label="VaR-Compliant (Empirical)")
plt.scatter(np.array(volatilities)[~empirical_VaR_compliant], np.array(mu_values)[~empirical_VaR_compliant], color='red', label="VaR-Violating (Empirical)")
plt.scatter(np.array(volatilities)[discrepancy], np.array(mu_values)[discrepancy], color='blue', label="Discrepancy")
plt.xlabel("Volatility")
plt.ylabel("Expected Return")
# plt.title("Efficient Frontier with Confidence Intervals and Empirical VaR Compliance")
plt.legend()
plt.grid()
plt.savefig('Plots/Efficient_Frontier.pdf', format='pdf')
plt.show()

#%%
# Bob's returns (median target return portfolio)
bob_portfolio_index = len(mu_values) // 2  # Median portfolio
bob_returns = returns_simulated @ weights[bob_portfolio_index]

# Compute 5% VaR for Bob's returns
bob_VaR_5 = np.percentile(bob_returns, 5)

# Plot Bob's returns distribution
plt.figure(figsize=(10, 6))
kde = gaussian_kde(bob_returns)
x_grid = np.linspace(np.min(bob_returns), np.max(bob_returns), 1000)
plt.plot(x_grid, kde(x_grid), label="Bob's Returns")

# Add vertical line for 5% VaR
plt.axvline(x=bob_VaR_5, color='red', linestyle='--', label="5% VaR")

plt.xlabel("Portfolio Return")
plt.ylabel("Density")
plt.title("Distribution of Bob's Returns with 5% VaR")
plt.legend()
plt.grid()
plt.show()
# %%
# Alice's returns (minimum target return portfolio)
alice_portfolio_index = 0  # Minimum target return portfolio
alice_returns = returns_simulated @ weights[alice_portfolio_index]

# Compute 5% VaR for Alice's returns
alice_VaR_5 = np.percentile(alice_returns, 5)

# Plot Alice's returns distribution
plt.figure(figsize=(10, 6))
kde = gaussian_kde(alice_returns)
x_grid = np.linspace(np.min(alice_returns), np.max(alice_returns), 1000)
plt.plot(x_grid, kde(x_grid), label="Alice's Returns")

# Add vertical line for 5% VaR
plt.axvline(x=alice_VaR_5, color='red', linestyle='--', label="5% VaR")

plt.xlabel("Portfolio Return")
plt.ylabel("Density")
plt.title("Distribution of Alice's Returns with 5% VaR")
plt.legend()
plt.grid()
plt.show()
# %%
# Plot Both of them on one plot
# Bob's returns (median target return portfolio)
bob_portfolio_index = len(mu_values) // 2  # Median portfolio
bob_returns = returns_simulated @ weights[bob_portfolio_index]
bob_VaR_70 = np.percentile(bob_returns, 30)  # 30th percentile for λ = 0.70

# Alice's returns (minimum target return portfolio)
alice_portfolio_index = 0  # Minimum target return portfolio
# alice_returns = returns_simulated @ weights[alice_portfolio_index]
alice_returns = mu_values
alice_VaR_70 = np.percentile(alice_returns, 30)  # 30th percentile for λ = 0.70

# Define a common grid for the x-axis
x_grid = np.linspace(min(np.min(bob_returns), np.min(alice_returns)), 
                     max(np.max(bob_returns), np.max(alice_returns)), 
                     1000)

# Compute KDE for Bob's and Alice's returns
kde_bob = gaussian_kde(bob_returns)
kde_alice = gaussian_kde(alice_returns)

# Plot both distributions
plt.figure(figsize=(10, 6))
plt.plot(x_grid, kde_bob(x_grid), label="Bob's Returns", color='blue')
plt.plot(x_grid, kde_alice(x_grid), label="Alice's Returns", color='green')

# # Add vertical lines for VaR at λ = 0.70
# plt.axvline(x=bob_VaR_70, color='red', linestyle='--', label="Bob's VaR (λ = 0.70)")
# plt.axvline(x=alice_VaR_70, color='orange', linestyle='--', label="Alice's VaR (λ = 0.70)")

# Add labels, title, and legend
plt.xlabel("Portfolio Return")
plt.ylabel("Density")
# plt.title("Distribution of Bob's and Alice's Returns with VaR (λ = 0.70)")
plt.legend()
plt.grid()
plt.savefig("Plots/return_distribution.pdf", format="pdf")
plt.show()


# %%
