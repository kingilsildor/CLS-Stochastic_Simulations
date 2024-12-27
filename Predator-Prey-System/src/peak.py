# peak & trough
from scipy.signal import find_peaks
import numpy as np
import importantce_check as ic
from optimization_algo import simulated_annealing as sa 
import weighted_objective
import matplotlib.pyplot as plt
import fft_objective as fft_obj


# load the data from other python files
def load_data():
        data = np.loadtxt("../observed_data/predator-prey-data.csv", delimiter=",")
        return data[:, 0], data[:, 1], data[:, 2]
    
raw_data = load_data()
t_data = raw_data[0]
x_data = raw_data[1]
y_data = raw_data[2]

global_ODE_param = [0.84321067, 0.426050229, 1.16945457, 2.03212881]
global_SA_param = [
        np.array([1.0, 0.1, 0.1, 1.0]),  # initial_state
        [(0.1, 1), (0.1, 1), (0.1, 2), (0.1, 3)],  # bounds
        lambda x: x + np.random.normal(0, 0.1, size=len(x)),  # proposal_func
        100,  # initial_temp
        0.99,  # alpha
        500,  # max_iter, markov chain length
    ]
initial_state, bounds, proposal_func, initial_temp, alpha, max_iter = global_SA_param

import_x = [0.06710567, 0.19507186, 3.23985999, 0.1386161,  6.36570217, 3.23202922,
    0.08388873, 3.09812684, 0.15495012 ,3.23215634, 0.05185563, 0.2817658,
    3.15827282, 0.06365766, 3.15986102 ,3.35004225, 3.12343857, 0.14891486,
    3.31781091, 3.26101711, 3.12251972 ,0.08714575, 6.32226904, 3.005135,
    6.17353856, 6.27222641, 3.03424178 ,3.22398865, 0.83766613, 3.06263755,
    3.07589764, 3.02405844, 3.08460424 ,6.27393723, 3.18354473, 0.10144565,
    3.07897569, 0.03360907, 0.13520543, 3.23974399, 3.25745307, 4.12182831,
    0.12958525, 0.25927301 ,3.21074874 ,3.31464777, 0.20940894, 6.22132361,
    0.20084509, 0.11115314, 5.38719762 ,3.13827159, 3.23922999, 3.15950523,
    0.16770393, 0.04606243, 3.08533693 ,3.1987016,  3.2899426,  3.89018743,
    0.26316879, 3.25160443, 6.09216614 ,6.3563936,  3.29184116, 0.09632277,
    0.62855249 ,0.13120175, 6.23270618 ,3.28855957, 3.19020723, 6.43162523,
    3.07880616, 3.037464  , 3.07356786, 3.24797379, 0.09889848, 3.16804085,
    6.23550392, 3.31193518, 5.32793263, 3.28862847, 3.26743309, 6.38322911,
    6.26550936, 3.19078069 ,0.11104114, 3.25193169, 3.26248027, 3.03418743,
    6.30398377, 3.26811224 ,3.30496062 ,3.24764021, 6.22765289, 3.1213283,
    0.33342429, 6.3868726 , 3.23100789, 1.94877788]
import_y =  [3.3429743,  0.13103341, 3.26386014, 6.10267342, 0.13477517, 0.10717735,
    0.67184043, 6.09988026, 0.1576729,  0.10789452, 3.30422263, 0.14155401,
    0.07161261 ,3.02355962, 0.09971401, 6.33244131, 3.17565855, 6.40553236,
    6.17466189 ,6.33440018, 3.26180286, 0.13505771, 6.32810228, 3.30170823,
    3.1075047  ,3.3381902 , 6.20943094, 3.50330669, 3.25940823, 3.13368048,
    3.27401205 ,0.1836304 , 3.26353402, 4.63990877, 0.20510497, 3.16202071,
    3.21282786 ,0.0694859 , 6.19797663 ,0.10605501, 3.23663232, 3.07344717,
    0.11301269 ,6.13920848, 3.3685336  ,3.09987176, 3.04057022, 6.4687988,
    3.09423741 ,6.36560324 ,3.23917211 ,1.02853817, 3.35206863, 6.07774304,
    3.24104718 ,0.83491118, 6.29390083 ,3.19741791, 6.24306857, 3.17706214,
    3.3241074  ,6.31356853, 3.13553537 ,0.21161276, 3.23622737, 6.30691142,
    3.1188802  ,6.3918742 , 0.50730425 ,6.22983498, 0.13784034, 3.30371674,
    6.35615831 ,3.24529634, 3.12823302 ,0.03884129, 6.17685171, 3.27339587,
    6.25903412 ,0.07024611, 3.22717155 ,6.23099972, 0.14889149, 3.1892271,
    6.07519259 ,0.14619932, 3.29369482 ,3.22204459, 3.29067816, 3.15957082,
    6.20887812 ,2.11484691, 6.34740603 ,3.19095153, 6.06535949, 3.08968063,
    3.20953152 ,3.18339847, 3.31668822 ,0.18036691]

# x_data = np.array(import_x)
# y_data = np.array(import_y)

# Detect peaks and troughs for predator and prey
peaks_x, _ = find_peaks(x_data, height= 3.5)
troughs_x, _ = find_peaks(-x_data, height= -1)


peaks_y, _ = find_peaks(y_data, height= 5.5)
troughs_y, _ = find_peaks(-y_data, height = -1)

print("Peaks in X:", peaks_x)
print("Troughs in X:", troughs_x)
print("Peaks in Y:", peaks_y)
print("Troughs in Y:", troughs_y)

# regard peak and trough as the critical indices
critical_indices_x = sorted(list(peaks_x) + list(troughs_x))
critical_indices_y = sorted(list(peaks_y) + list(troughs_y))

# remove points around the critical indices
def remove_points_around_indices(data, critical_indices, window_size):
    indices_to_remove = [idx + i for idx in critical_indices for i in range(-window_size, window_size + 1) if 0 <= idx + i < len(data)]
    return sorted(indices_to_remove)

# change different window sizes to see the difference in results
window_size = 2
# Separate critical indices for X and Y
removed_indices_x = remove_points_around_indices(x_data, critical_indices_x, window_size)
removed_indices_y = remove_points_around_indices(y_data, critical_indices_y, window_size)

# Combine indices to modify all datasets consistently
# combined_indices = sorted(set(removed_indices_x).union(set(removed_indices_y)))
combined_indices = removed_indices_x
x_data_modified = np.delete(x_data, combined_indices)
y_data_modified = np.delete(y_data, combined_indices)
t_data_modified = np.delete(t_data, combined_indices)

# Create trained data with separate zero indices for X and Y
trained_data = (x_data_modified, y_data_modified, t_data_modified, set(removed_indices_x), set())
print(f'trained data {trained_data}')
# Perform simulated annealing
print(type(weighted_objective.objective_weighted))
weighted_params, weighted_mse = ic.simulated_annealing_for_importance_check(weighted_objective.objective_weighted, trained_data, initial_state, bounds, proposal_func, initial_temp, alpha, max_iter)

# Compute impact on fit
mse_global = ic.global_objective(global_ODE_param, t_data, x_data, y_data)
mse_weighted = ic.global_objective(weighted_params, t_data, x_data, y_data)
impact_on_fit = abs(mse_global - mse_weighted)
print("Impact on Fit (MSE Difference):", impact_on_fit)

# Compute parameter differences
param_diff = np.abs(np.array(global_ODE_param) - np.array(weighted_params))
print("Parameter Changes:", param_diff)


