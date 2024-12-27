import matplotlib.pyplot as plt
import file_sys as fs
import numpy as np

def plot_pair_waiting_time_diff(n0, n1, rho, waiting_times_diff, filename_prefix):
    """
    Plot the difference in waiting times for customers in two systems

    Parameters:
    ----------
    n0: int
        number of servers in the first system
    n1: int
        number of servers in the second system
    rho: float
        traffic intensity
    waiting_times_diff: list
        list of differences in waiting times
    filename_prefix: str
        prefix of the filename
    """
    plt.figure(figsize=(10, 6))
    plt.xlabel("Customer Number")
    plt.ylabel("Waiting Time Difference")
    plt.title(f"Difference in Waiting Time for Customers in Two Systems, with $n=${n0} and $n=${n1} servers")
    plt.plot(waiting_times_diff, label=f"$n=${n0} - $n=${n1}")
    plt.legend()
    plt.grid()

    # check if the directory exists
    if not fs.check_directory(fs.SIMU_VISUALIZATION_PATH):
        fs.create_directory(fs.SIMU_VISUALIZATION_PATH)
    plt.savefig(f"{fs.SIMU_VISUALIZATION_PATH}{filename_prefix}_waiting_time_diff_with_n_{n0}_and_{n1}_rho_{rho}.png")

def visulize_all_parameters_pair_diff_waiting_time(n0, n1, mu, lambdas, fileName_prefix):
    """
    Visualize the difference in waiting times for customers in two systems

    Parameters:
    ----------
    n0: int
        number of servers in the first system
    n1: int
        number of servers in the second system
    mu: float
        service rate
    lambdas: list
        list of arrival rates
    fileName_prefix: str
        prefix of the filename
    """
    i = 0
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'h', 'H', '8', '*', 'X']
    results = {}
    for lam in lambdas:
        _, _, waiting_time_base, _ = fs.read(n0, fs.SIMU_RESULT_PATH, lam, mu)
        _, _, waiting_time_compared, _ = fs.read(n1, fs.SIMU_RESULT_PATH, lam, mu)
        rho = lam / mu
        waiting_time_compared = waiting_time_compared[:len(waiting_time_base)]
        results[f"$\\rho=${rho}"] = np.array(waiting_time_base) - np.array(waiting_time_compared)

    min_len = min([len(value) for value in results.values()])
    for key, value in results.items():
        results[key] = value[:min_len]

    plt.figure(figsize=(10, 6))
    plt.xlabel("Customer Number")
    plt.ylabel("Waiting Time Difference")
    plt.title(f"Difference in Waiting Time for Customers in Two Systems, with $n=${n0} and $n=${n1} servers")

    for key, value in results.items():
        plt.plot(value, label=key, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
        i += 1
    plt.legend()
    plt.grid()
    plt.savefig(f"{fs.SIMU_VISUALIZATION_PATH}{fileName_prefix}_waiting_time_diff_with_n_{n0}_and_{n1}.png")

    for lam in lambdas:
        _, _, waiting_time_base, _ = fs.read(n0, fs.SIMU_RESULT_PATH, lam, mu)
        _, _, waiting_time_compared, _ = fs.read(n1, fs.SIMU_RESULT_PATH, lam, mu)
        _, _, waiting_time_compare4, _ = fs.read(4, fs.SIMU_RESULT_PATH, lam, mu)
        rho = lam / mu
        plt.figure(figsize=(10, 6))
        plt.xlabel("Customer Number", fontsize=20)
        plt.ylabel("Waiting Time", fontsize=20)
        plt.title(f"Waiting Time for Customers in 3 Systems, with $n=${n0}, $n=${n1} and $n=4$ servers, $\\rho=${rho}",fontsize=16)
        plt.plot(waiting_time_base, label=f"n={n0}", color='b', linestyle='-')
        plt.plot(waiting_time_compared, label=f"n={n1}", color='r', linestyle='-.')
        plt.plot(waiting_time_compare4, label="n=4", color='g', linestyle='--')
        plt.tick_params(axis='both', labelsize=14)
        
        plt.legend()
        plt.grid()
        plt.savefig(f"{fs.SIMU_VISUALIZATION_PATH}{fileName_prefix}_waiting_time_compare_with_n_{n0}_{n1}_and_4_rho_{rho}.png")


def visulize_all_parameters_pair_diff_waiting_time_sjf(mu, lambdas, fileName_prefix):
    """
    Visualize the difference in waiting times for customers in SJF and FIFO systems

    Parameters:
    ----------
    mu: float
        service rate
    lambdas: list
        list of arrival rates
    fileName_prefix: str
        prefix of the filename
    """
    i = 0
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'h', 'H', '8', '*', 'X']
    results = {}
    
    for lam in lambdas:
        customers_base, _, waiting_time_base, _= fs.read(1, fs.SIMU_RESULT_PATH, lam, mu, prefix="Comparison_FIFO")
        customers_compared, _, waiting_time_compared, _ = fs.read(1, fs.SIMU_RESULT_PATH, lam, mu, prefix="Comparison_SJF")
        sorted_data = sorted(zip(customers_compared, waiting_time_compared), key=lambda x: x[0])
        customers_compared, waiting_time_compared = zip(*sorted_data)
        customers_compared = list(customers_compared)
        waiting_time_compared = list(waiting_time_compared)

        rho = lam / mu
        

        plt.figure(figsize=(10, 6))

        plt.xlabel("Customer Number", fontsize=20)
        plt.ylabel("Waiting Time", fontsize=20)
        plt.title(f"Waiting Time for Customers in SFJ and FIFO Systems, both with $n=1$ server", fontsize=16)

        plt.tick_params(axis='x', labelsize=12)  # set X axis scale font size
        plt.tick_params(axis='y', labelsize=14)  # set Y axis scale font size

        plt.plot(customers_base, waiting_time_base, label=f"FIFO_$\\rho=${rho}", color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
        i += 1
        plt.plot(customers_compared, waiting_time_compared, label=f"SJF_$\\rho=${rho}", color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])

        # calculate the area under each plot
        area_base = np.trapz(waiting_time_base, customers_base)
        area_compared = np.trapz(waiting_time_compared, customers_compared)
        print(f"rho: {rho}, area_base: {area_base}, area_compared: {area_compared}, diff: {area_base - area_compared}")
            
        plt.legend(fontsize=12)
        plt.grid()
        plt.savefig(f"{fs.SIMU_VISUALIZATION_PATH}{fileName_prefix}_waiting_time_diff_with_n_1_Rho_{ rho }.png")
        plt.close()

# Example Usage:
if __name__ == "__main__":
    visulize_all_parameters_pair_diff_waiting_time_sjf(1.0, [0.9], "SJF")