import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings
import pandas as pd
import seaborn as sns
import time

from scipy.integrate import odeint
from scipy import stats
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool, Lock
from optimization_algo import simulated_annealing

lock = Lock()
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.integrate")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_data():
    """
    Load data from path, csv file
    """
    data = np.loadtxt("../data/predator-prey-data.csv", delimiter=",")
    return data[:, 0], data[:, 1], data[:, 2]


def calculate_mse(data):
    """
    Calculate the Mean Squared Error (MSE) for each combination of parameters
    Returns a DataFrame with the MSE values
    """
    df = data.groupby(["Alpha", "Temperature", "Max_Iter"])["MSE"].mean().reset_index()
    df["Alpha"] = np.round(df["Alpha"], 3)
    df["Max_Iter"] = df["Max_Iter"].astype(int)
    return df


def calc_stats_mse_iter(data):
    """
    Calculate the mean, standard deviation and variance of the MSE for each Max_Iter value
    Returns a DataFrame with the statistics
    """

    df = data.groupby(["Max_Iter"])["MSE"].agg(["mean", "std", "var"]).reset_index()
    return df


def lotka_volterra(y, t, alpha, beta, delta, gamma):
    """
    Lotka-Volterra equations (predator-prey model)
    """
    x, y = y
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]


def proposal_func(x):
    """
    Symetric proposal function to generate a new state
    """
    return x + np.random.normal(0, 0.1, size=x.shape)


def objective(params, t_data, x_data, y_data):
    """
    Objective function to minimize (sum of squared errors)
    """
    alpha, beta, delta, gamma = params
    y0 = [x_data[0], y_data[0]]  # Initial condition
    solution = odeint(lotka_volterra, y0, t_data, args=(alpha, beta, delta, gamma))
    x_sim, y_sim = solution[:, 0], solution[:, 1]
    mse = np.mean((x_sim - x_data) ** 2 + (y_sim - y_data) ** 2)  # Mean Squared Error
    return mse


def compute_model(t_data, x_data, y_data, alpha, beta, delta, gamma):
    """
    Compute the Lotka-Volterra model using the best parameters
    """
    y0 = [x_data[0], y_data[0]]
    solution = odeint(lotka_volterra, y0, t_data, args=(alpha, beta, delta, gamma))
    prey = solution[:, 0]
    predator = solution[:, 1]
    return predator, prey


def get_N_best_params(data, N=10):
    """
    Return the best parameters for the Lotka-Volterra model
    """
    return data.nsmallest(N, "MSE")


def get_best_params(data):
    """
    Return the best parameters for the Lotka-Volterra model
    """
    return data.nsmallest(1, "MSE")


def process_combination(args):
    """
    Process a single combination of parameters.
    """
    (
        alpha,
        temperature,
        max_iter,
        N_boots,
        data,
        params,
        bounds,
        objective,
        proposal_func,
    ) = args
    results = []

    for _ in range(N_boots):
        best_params, best_mse = simulated_annealing(
            objective,
            data,
            params,
            bounds,
            proposal_func,
            temperature,
            alpha,
            int(max_iter),
        )
        param_alpha, param_beta, param_delta, param_gamma = best_params
        results.append(
            (
                alpha,
                temperature,
                max_iter,
                best_mse,
                param_alpha,
                param_beta,
                param_delta,
                param_gamma,
            )
        )

    return results


def param_estimation(
    steps,
    alpha_bounds,
    temperature_bounds,
    N_boots,
    data,
    params,
    bounds,
    max_iter,
):
    """
    Perform parameter estimation using simulated annealing
    """
    alpha_list = np.linspace(*alpha_bounds, steps)
    temperature_list = np.linspace(*temperature_bounds, steps)
    iter_list = np.linspace(*max_iter, steps)

    combinations = list(product(alpha_list, temperature_list, iter_list))
    total_iterations = len(combinations) * N_boots

    argument_pairs = [
        (
            alpha,
            temperature,
            max_iter,
            N_boots,
            data,
            params,
            bounds,
            objective,
            proposal_func,
        )
        for alpha, temperature, max_iter in combinations
    ]

    with open("../data/full_param_estimation.csv", "a") as file:
        if file.tell() == 0:
            file.write(
                "Alpha,Temperature,Max_Iter,MSE,Param_Alpha,Param_Beta,Param_Delta,Param_Gamma\n"
            )

        with Pool() as pool:
            with tqdm(
                total=total_iterations, desc="Processing", unit="iteration"
            ) as pbar:
                for result_set in pool.imap_unordered(
                    process_combination, argument_pairs
                ):
                    with lock:  # Ensure only one process writes at a time
                        for result in result_set:
                            file.write(",".join(map(str, result)) + "\n")
                    pbar.update(len(result_set))


def test_param_estimation(data_partialy, data_full):
    """
    Perform a t-test to compare the MSE of the full and partial datasets
    """
    mse_partialy = data_partialy["MSE"]
    mse_full = data_full["MSE"]

    t_stat, p_value = stats.ttest_ind(mse_partialy, mse_full, equal_var=False)
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("The MSE of dataset 1 is significantly lower than dataset 2.")
    else:
        print("There is no significant difference in MSE between the two datasets.")


def plot_observed_data():
    """
    Plot the observed data
    """
    t_data, x_data, y_data = load_data()

    plt.figure(figsize=(18, 6), dpi=300)
    font_size = 16

    plt.plot(t_data, x_data, "o", label="Predator")
    plt.plot(t_data, y_data, "o", label="Prey")

    plt.xlabel("Time", fontsize=font_size + 2)
    plt.tick_params(axis="both", which="major", labelsize=font_size)
    plt.ylabel("Population", fontsize=font_size + 2)
    plt.title(
        "Generated data using Lotka-Volterra equations and Gaussian noise",
        fontsize=font_size + 4,
    )

    plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.savefig("../visualization/lotka_volterra_data.png")


def plot_model(t_data, x_data, y_data, sim_data, params):
    """
    Plot the observed data and the best fit
    """
    sim_predator, sim_prey = sim_data
    alpha, beta, delta, gamma = params

    plt.figure(figsize=(18, 6), dpi=300)
    font_size = 16

    plt.plot(t_data, x_data, label="Predator")
    plt.plot(t_data, y_data, label="Prey")
    plt.scatter(t_data, sim_prey, alpha=0.7)
    plt.scatter(t_data, sim_predator, alpha=0.7)

    black_dot = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        markersize=8,
        label="Simulated Data",
    )

    plt.xlabel("Time", fontsize=font_size + 2)
    plt.tick_params(axis="both", which="major", labelsize=font_size)
    plt.ylabel("Population", fontsize=font_size + 2)
    plt.title(
        f"Lotka-Volterra model with best parameters\n $\\alpha$={alpha:.5f}, $\\beta$={beta:.5f}, $\\delta$={delta:.5f}, $\\gamma$={gamma:.5f}",
        fontsize=font_size + 4,
    )

    plt.legend(
        fontsize=font_size,
        handles=plt.gca().get_legend_handles_labels()[0] + [black_dot],
    )
    plt.tight_layout()
    plt.savefig("../visualization/lotka_volterra_model.png")


def plot_mse_iter(data, name):
    """
    Plot the Mean Squared Error (MSE) vs Max Iterations
    """

    df = calc_stats_mse_iter(data)

    plt.figure(figsize=(18, 6), dpi=300)
    font_size = 16

    plt.errorbar(
        df["Max_Iter"],
        df["mean"],
        yerr=df["std"],
        fmt="-o",
        capsize=5,
        label=r"Mean $\pm \sigma$",
    )

    plt.xlabel(r"Max Iterations ($k$)", fontsize=font_size + 2)
    plt.xticks(df["Max_Iter"])
    plt.ylabel("Mean MSE", fontsize=font_size + 2)
    plt.title(r"Mean MSE vs Max Iterations ($k$)", fontsize=font_size + 4)
    plt.tick_params(axis="both", which="major", labelsize=font_size)

    plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.savefig(f"../visualization/max_iterations_{name}.png")


def plot_heatmap(data, name):
    """
    Plot a heatmap of the Mean Squared Error (MSE) for each combination of parameters
    """

    mean_mse = calculate_mse(data)
    pivot_table = mean_mse.pivot(index="Alpha", columns="Temperature", values="MSE")

    plt.figure(figsize=(18, 6), dpi=300)
    font_size = 16

    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap="viridis_r",
        cbar_kws={"label": "Mean MSE"},
        annot_kws={"size": font_size},
    )
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=font_size + 2)
    colorbar.set_label("MSE", fontsize=font_size + 2, labelpad=15)

    if name == "full":
        plt.title(
            "Mean MSE for $\\alpha$ and Temperature\n Using the full dataset",
            fontsize=font_size + 4,
        )
    else:
        plt.title(
            "Mean MSE for $\\alpha$ and Temperature\n Using the partial dataset",
            fontsize=font_size + 4,
        )
    plt.xlabel("Temperature", fontsize=font_size + 2)
    plt.ylabel(r"$\alpha$", fontsize=font_size + 2)
    plt.tick_params(axis="both", which="major", labelsize=font_size, labelrotation=0.45)

    plt.tight_layout()
    plt.savefig(f"../visualization/mse_heatmap_{name}.png")


if __name__ == "__main__":
    true_data = load_data()
    partualy_data = tuple(tuple(arr[::2]) for arr in true_data)
    time_array = true_data[0]

    estimated_param = pd.read_csv("../data/param_estimation.csv")
    estimated_param_full = pd.read_csv("../data/full_param_estimation.csv")
    data_partialy = estimated_param[estimated_param["Max_Iter"] == 800]
    data_full = estimated_param_full[estimated_param_full["Max_Iter"] == 800]

    initial_params = np.array(
        [1.0, 0.1, 0.1, 1.0]
    )  # Initial guess for [alpha, beta, delta, gamma]
    param_bounds = np.array(
        [
            (0.01, 3.0),
            (0.01, 3.0),
            (0.01, 3.0),
            (0.01, 3.0),
        ]
    )  # Parameter bounds

    start_time = time.time()
    # Original parameters should be steps:10, N_boots:40.
    param_estimation(
        steps=2,
        alpha_bounds=(0.8, 0.99),
        temperature_bounds=(10, 100),
        N_boots=10,
        data=partualy_data,
        params=initial_params,
        bounds=param_bounds,
        max_iter=(100, 1000),
    )
    print(f"Time taken: {time.time() - start_time}")

    plot_observed_data()
    plot_heatmap(data_partialy, "partial")
    plot_heatmap(data_full, "full")
    plot_mse_iter(estimated_param, "partial")
    plot_mse_iter(estimated_param_full, "full")

    test_param_estimation(estimated_param, estimated_param_full)

    best_params = (
        get_best_params(estimated_param)[
            ["Param_Alpha", "Param_Beta", "Param_Delta", "Param_Gamma"]
        ]
        .to_numpy()
        .flatten()
    )

    sim_data = compute_model(*true_data, *best_params)
    plot_model(*true_data, sim_data, best_params)
