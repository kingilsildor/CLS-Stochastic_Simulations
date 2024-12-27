import numpy as np
import scipy.stats as stats
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import os

IMG_STATISTIC_DIR = '../images/statistic_analysis'

# Load Mandelbrot area data from files
def load_area_data(file_path):
    areas = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if 'trueArea' in file_path:
                    # Extract the true area value from the line
                    areas.append(float(line.split()[-1]))
                else:
                    # Extract the area value from Mandelbrot area files
                    _, _, area = line.split()
                    areas.append(float(area))
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    return np.array(areas)

# Calculate the mean and variance of the Mandelbrot area results
def calculate_mean_and_variance():
    pure_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_Pure.txt')
    lhs_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_LHS.txt')
    ortho_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_Ortho.txt')

    mean_pure = np.mean(pure_areas)
    var_pure = np.var(pure_areas)

    mean_lhs = np.mean(lhs_areas)
    var_lhs = np.var(lhs_areas)

    mean_ortho = np.mean(ortho_areas)
    var_ortho = np.var(ortho_areas)

    return {
        'Pure': {'mean': mean_pure, 'variance': var_pure},
        'LHS': {'mean': mean_lhs, 'variance': var_lhs},
        'Ortho': {'mean': mean_ortho, 'variance': var_ortho}
    }

# Calculate the Mean Squared Error (MSE) of each area result compared to the true area
def calculate_mse():
    true_area = load_area_data(f'{utils.RESULT_DIR}/trueArea.txt')

    pure_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_Pure.txt')
    lhs_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_LHS.txt')
    ortho_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_Ortho.txt')

    mse_pure = np.mean((pure_areas - true_area) ** 2)
    mse_lhs = np.mean((lhs_areas - true_area) ** 2)
    mse_ortho = np.mean((ortho_areas - true_area) ** 2)

    return {
        'Pure': mse_pure,
        'LHS': mse_lhs,
        'Ortho': mse_ortho
    }

# Calculate confidence intervals and determine if they include the true area
def calculate_confidence_intervals():
    true_area = load_area_data(f'{utils.RESULT_DIR}/trueArea.txt')

    pure_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_Pure.txt')
    lhs_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_LHS.txt')
    ortho_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_Ortho.txt')

    confidence_level = 0.95
    # calculate the z-value for the confidence level
    z_value = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Helper function to calculate the confidence interval for the given areas
    def calculate_interval(areas):
        mean = np.mean(areas)
        std = np.std(areas, ddof=1)
        n = len(areas)

        t_critical = stats.t.ppf(confidence_level/ 2, n - 1)
        margin_of_error = t_critical * (std / np.sqrt(n))

        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error

        if lower_bound <= true_area <= upper_bound:
            includes_true_area = True
        else:
            includes_true_area = False

        return {
            'mean': mean,
            'standard_deviation': std,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            't_critical': t_critical,
            'true_area_included': includes_true_area
        }

    pure_interval = calculate_interval(pure_areas)
    lhs_interval = calculate_interval(lhs_areas)
    ortho_interval = calculate_interval(ortho_areas)

    intervals = {
        'Pure interval': pure_interval, 
        'LHS interval': lhs_interval, 
        'Ortho inteval': ortho_interval
    } 
    
    return intervals

# Plot the confidence intervals for the Mandelbrot area results
def plot_histograms():
    pure_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_Pure.txt')
    lhs_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_LHS.txt')
    ortho_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_Ortho.txt')
    true_value = load_area_data(f'{utils.RESULT_DIR}/trueArea.txt')[0]
    intervals = calculate_confidence_intervals().values()

    areas = [pure_areas, lhs_areas, ortho_areas]
    labels = ['Pure', 'LHS', 'Ortho']
    max_arr_len = max(len(area) for area in areas)
    min_area = min(min(area) for area in areas)
    max_area = max(max(area) for area in areas)
    bin_size = int(np.sqrt(max_arr_len))

    for area, label, inteval in zip(areas, labels, intervals):
        mean_area = np.mean(area)
        bin_edges = np.linspace(min_area, max_area, bin_size + 1)

        plt.hist(area, bins=bin_edges, alpha=0.6)

        lower_bound = inteval['lower_bound']
        upper_bound = inteval['upper_bound']

        plt.xlabel('Area')
        plt.ylabel('Frequency')
        plt.axvline(mean_area, color='k', linestyle='dashed', linewidth=1, label=f'Mean Area: {mean_area:.6f}')
        plt.axvline(true_value, color='b', linestyle='dashed', linewidth=1, label=f'True Area: {true_value:.6f}')

        plt.axvline(lower_bound, color='r', linewidth=1, label=f'Lower Bound: {lower_bound:.6f}')
        plt.axvline(upper_bound, color='r', linewidth=1, label=f'Upper Bound: {upper_bound:.6f}')

        plt.title(f'{label} Histogram')
        # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.xlim(min_area, max_area)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_STATISTIC_DIR, f'histogram_{label}.png'))
        plt.close()

# Plot the confidence intervals for the Mandelbrot area results
def plot_confidence_intervals(intervals):
    pure_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_Pure.txt')
    lhs_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_LHS.txt')
    ortho_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_Ortho.txt')
    areas = [pure_areas, lhs_areas, ortho_areas]

    labels = ['Pure', 'LHS', 'Ortho']

    for interval, label, area in zip(intervals.values(), labels, areas):
        mean = interval['mean']
        std_dev = interval['standard_deviation']

        kde = stats.gaussian_kde(area)
        x_vals = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
        y_vals = kde(x_vals)
        
        mode_index = np.argmax(y_vals)
        mean = x_vals[mode_index]
        
        plt.plot(x_vals, y_vals, color='b')
        plt.fill_between(x_vals, y_vals, color='b', alpha=0.1)

        colors = ["red", "orange", "green", "black", "green", "orange", "red"]
        for i, color in zip(range(-3, 4), colors):
            x_line = mean + i * std_dev
            y_value = kde(x_line)

            mean_label = None
            if i == 0:
                mean_label = f'Mean: {mean:.6f}'
            plt.vlines(x_line, ymin=0, ymax=y_value, color=color, linestyle=':', label=mean_label)

        plt.xlabel('Area')
        plt.ylabel('Density')
        plt.title(f'{label} Confidence Interval')
        plt.legend()
        plt.savefig(os.path.join(IMG_STATISTIC_DIR, f'{label}_CI.png'))
        plt.close()

# Plot the distributions of the Mandelbrot area results
def plot_area_distributions():
    pure_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_Pure.txt')
    lhs_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_LHS.txt')
    ortho_areas = load_area_data(f'{utils.STATISTIC_RESULT_DIR}/mandelbrotArea_Ortho.txt')

    data = [pure_areas, lhs_areas, ortho_areas]
    labels = ['Pure', 'LHS', 'Ortho']

    # Boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels)
    plt.title('Boxplot of Mandelbrot Area Results')
    plt.ylabel('Area')
    plt.savefig(os.path.join(IMG_STATISTIC_DIR, 'boxplot_mandelbrot_area.png'))
    plt.close()

    # Violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=data)
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.title('Violin Plot of Mandelbrot Area Results')
    plt.ylabel('Area')
    plt.savefig(os.path.join(IMG_STATISTIC_DIR, 'violinplot_mandelbrot_area.png'))
    plt.close()

    # Histogram
    plt.figure(figsize=(8, 6))
    for i, area_data in enumerate(data):
        sns.histplot(area_data, kde=True, label=labels[i], bins=20, alpha=0.6)
    plt.legend()
    plt.title('Histogram of Mandelbrot Area Results')
    plt.xlabel('Area')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(IMG_STATISTIC_DIR, 'histogram_mandelbrot_area.png'))
    plt.close()

    # Strip plot
    plt.figure(figsize=(8, 6))
    sns.stripplot(data=data, jitter=True)
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.title('Strip Plot of Mandelbrot Area Results')
    plt.ylabel('Area')
    plt.savefig(os.path.join(IMG_STATISTIC_DIR, 'stripplot_mandelbrot_area.png'))
    plt.close()

# Example usage
if __name__ == "__main__":
    mean_and_variance = calculate_mean_and_variance()
    # print("Mean and Variance:", mean_and_variance)

    mse = calculate_mse()
    # print("Mean Squared Error (MSE):", mse)

    confidence_intervals = calculate_confidence_intervals()
    plot_confidence_intervals(confidence_intervals['intervals'].values(),
                              confidence_intervals['areas'].values())
    # print("Confidence Intervals:", confidence_intervals)

    # Plot area distributions
    plot_area_distributions()