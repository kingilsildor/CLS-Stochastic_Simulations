from scipy.fft import fft
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from scipy.integrate import odeint
import numpy as np

def lotka_volterra(y, t, alpha, beta, delta, gamma):
    """
    Lotka-Volterra equations (predator-prey model)
    """
    x, y = y
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

def calculate_fft_similarity(x_data, y_data, sol, t):
    # fft of the data and the solution
    fft_data_prey = np.abs(fft(x_data))
    fft_data_predator = np.abs(fft(y_data))
    fft_sol_prey = np.abs(fft(sol[:, 0]))
    fft_sol_predator = np.abs(fft(sol[:, 1]))
    
    # calculate the difference between the two signals
    prey_spectrum_diff = euclidean(fft_data_prey[:len(t)//2], fft_sol_prey[:len(t)//2])
    predator_spectrum_diff = euclidean(fft_data_predator[:len(t)//2], fft_sol_predator[:len(t)//2])
    
    # calculate the correlation between the two signals
    prey_corr, _ = pearsonr(fft_data_prey[:len(t)//2], fft_sol_prey[:len(t)//2])
    predator_corr, _ = pearsonr(fft_data_predator[:len(t)//2], fft_sol_predator[:len(t)//2])
    
    return prey_spectrum_diff, predator_spectrum_diff, prey_corr, predator_corr

def objective_fft(params, t_data, x_data, y_data, zero_index_x, zero_index_y):
    alpha, beta, delta, gamma = params
    y0 = [x_data[0], y_data[0]]

    sol = odeint(lotka_volterra, y0, t_data, args=(alpha, beta, delta, gamma))
    mse_error = np.sqrt(np.mean((x_data[:] - sol[:, 0])**2 + (y_data[:] - sol[:, 1])**2))
    prey_spectrum_diff, predator_spectrum_diff, _, _ = calculate_fft_similarity(x_data, y_data, sol, t_data)

    return  mse_error + 0.025 * (prey_spectrum_diff + predator_spectrum_diff)  # weighted sum of errors
