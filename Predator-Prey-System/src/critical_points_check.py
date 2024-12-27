import numpy as np
import importantce_check as ic
import matplotlib.pyplot as plt
import fft_objective as fft_obj
from scipy.signal import find_peaks
import numpy as np
import importantce_check as ic

def find_XY_peaks(x_data, y_data):
    """
    Find peaks and troughs for predator and prey.

    Parameters:
    - x_data: array, predator data.
    - y_data: array, prey data.

    Returns:
    - peaks_x: array, indices of peaks in x_data.
    - troughs_x: array, indices of troughs in x_data.
    - peaks_y: array, indices of peaks in y_data.
    - troughs_y: array, indices of troughs in y_data.
    """
    peaks_x, _ = find_peaks(x_data, height= 3.5)
    troughs_x, _ = find_peaks(-x_data, height= -1)
    peaks_y, _ = find_peaks(y_data, height= 5.5)
    troughs_y, _ = find_peaks(-y_data, height = -1)
    return peaks_x, troughs_x, peaks_y, troughs_y


def critical_number_check(
    x_data,
    y_data,
    t_data,
    importance_X,
    importance_Y,
    weighted_objective,
    global_SA_param,
    global_objective,
    global_ODE_param,
    repeat=10,
):
    """
    Evaluate the effect of sequentially removing data points based on their importance.

    Parameters:
    - x_data: array, predator data.
    - y_data: array, prey data.
    - t_data: array, time data.
    - importance_X: list, importance values for x_data points.
    - importance_Y: list, importance values for y_data points.
    - weighted_objective: function, objective function with weights.
    - global_SA_param: SA parameters, [initial_state, bounds, proposal_func, initial_temp, alpha, max_iter].
    - global_objective: function, global objective function without weights.
    - global_ODE_param: list, globally optimized parameters [alpha, beta, delta, gamma].
    - repeat: int, number of times to repeat the SA algorithm.

    Returns:
    - x_removal_impact: list, impact on objective by removing y_data points while keeping x_data fixed.
    - y_removal_impact: list, impact on objective by removing x_data points while keeping y_data fixed.
    """
    # Sort indices by importance (ascending order)
    sorted_indices_X = np.argsort(importance_X)
    sorted_indices_Y = np.argsort(importance_Y)
    x_removal_impact = []
    y_removal_impact = []
    initial_state, bounds, proposal_func, initial_temp, alpha, max_iter = (
        global_SA_param
    )

    x_data_copy = x_data.copy()
    y_data_copy = y_data.copy()

    peaks_x, troughs_x, peaks_y, troughs_y = find_XY_peaks(x_data, y_data)

    critical_indices_x = sorted(list(peaks_x) + list(troughs_x))
    critical_indices_y = sorted(list(peaks_y) + list(troughs_y))

    # find rest indices for x and y
    rest_indices_x = [i for i in range(len(x_data)) if i not in critical_indices_x]
    rest_indices_y = [i for i in range(len(y_data)) if i not in critical_indices_y]

    new_ordered_x_indices = np.concatenate((np.array(critical_indices_x), np.array(rest_indices_x)))
    new_ordered_y_indices = np.concatenate((np.array(critical_indices_y), np.array(rest_indices_y)))

    for r in range(repeat):
        x_remove_impact_once = []
        y_remove_impact_once = []

        # Remove Y points and measure impact while keeping X fixed
<<<<<<< HEAD
        for remove_count in range(0, len(y_data), 1):
            zero_index_y = set(sorted_indices_Y[: remove_count + 1])
=======
        for remove_count in range(0, len(new_ordered_y_indices), 1):
            zero_index_y = set(new_ordered_y_indices[:remove_count + 1])
>>>>>>> refs/remotes/origin/main
            zero_index_x = set()  # No X points removed
            trained_data = (
                x_data,
                y_data,
                t_data,
                zero_index_x,
                zero_index_y,
            )  # Data to pass to SA

            weighted_params, weighted_mse = ic.simulated_annealing_for_importance_check(
                weighted_objective,
                trained_data,
                initial_state,
                bounds,
                proposal_func,
                initial_temp,
                alpha,
                max_iter,
            )
            mse_global = global_objective(global_ODE_param, t_data, x_data, y_data)
            mse_weighted = global_objective(weighted_params, t_data, x_data, y_data)
            x_remove_impact_once.append(abs(mse_global - mse_weighted))

        # Remove X points and measure impact while keeping Y fixed
<<<<<<< HEAD
        for remove_count in range(0, len(x_data), 1):
            zero_index_x = set(sorted_indices_X[: remove_count + 1])

=======
        for remove_count in range(0, len(new_ordered_x_indices), 1):
            zero_index_x = set(new_ordered_x_indices[:remove_count + 1])
>>>>>>> refs/remotes/origin/main
            zero_index_y = set()  # No Y points removed
            trained_data = (
                x_data,
                y_data,
                t_data,
                zero_index_x,
                zero_index_y,
            )  # Data to pass to SA

            weighted_params, weighted_mse = ic.simulated_annealing_for_importance_check(
                weighted_objective,
                trained_data,
                initial_state,
                bounds,
                proposal_func,
                initial_temp,
                alpha,
                max_iter,
            )
            mse_global = global_objective(global_ODE_param, t_data, x_data, y_data)
            mse_weighted = global_objective(weighted_params, t_data, x_data, y_data)
            y_remove_impact_once.append(abs(mse_global - mse_weighted))

        # Append impact values for this repeat
        x_removal_impact.append(x_remove_impact_once)
        y_removal_impact.append(y_remove_impact_once)

    return x_removal_impact, y_removal_impact


if __name__ == "__main__":
    import_x = [
        0.06710567,
        0.19507186,
        3.23985999,
        0.1386161,
        6.36570217,
        3.23202922,
        0.08388873,
        3.09812684,
        0.15495012,
        3.23215634,
        0.05185563,
        0.2817658,
        3.15827282,
        0.06365766,
        3.15986102,
        3.35004225,
        3.12343857,
        0.14891486,
        3.31781091,
        3.26101711,
        3.12251972,
        0.08714575,
        6.32226904,
        3.005135,
        6.17353856,
        6.27222641,
        3.03424178,
        3.22398865,
        0.83766613,
        3.06263755,
        3.07589764,
        3.02405844,
        3.08460424,
        6.27393723,
        3.18354473,
        0.10144565,
        3.07897569,
        0.03360907,
        0.13520543,
        3.23974399,
        3.25745307,
        4.12182831,
        0.12958525,
        0.25927301,
        3.21074874,
        3.31464777,
        0.20940894,
        6.22132361,
        0.20084509,
        0.11115314,
        5.38719762,
        3.13827159,
        3.23922999,
        3.15950523,
        0.16770393,
        0.04606243,
        3.08533693,
        3.1987016,
        3.2899426,
        3.89018743,
        0.26316879,
        3.25160443,
        6.09216614,
        6.3563936,
        3.29184116,
        0.09632277,
        0.62855249,
        0.13120175,
        6.23270618,
        3.28855957,
        3.19020723,
        6.43162523,
        3.07880616,
        3.037464,
        3.07356786,
        3.24797379,
        0.09889848,
        3.16804085,
        6.23550392,
        3.31193518,
        5.32793263,
        3.28862847,
        3.26743309,
        6.38322911,
        6.26550936,
        3.19078069,
        0.11104114,
        3.25193169,
        3.26248027,
        3.03418743,
        6.30398377,
        3.26811224,
        3.30496062,
        3.24764021,
        6.22765289,
        3.1213283,
        0.33342429,
        6.3868726,
        3.23100789,
        1.94877788,
    ]
    import_y = [
        3.3429743,
        0.13103341,
        3.26386014,
        6.10267342,
        0.13477517,
        0.10717735,
        0.67184043,
        6.09988026,
        0.1576729,
        0.10789452,
        3.30422263,
        0.14155401,
        0.07161261,
        3.02355962,
        0.09971401,
        6.33244131,
        3.17565855,
        6.40553236,
        6.17466189,
        6.33440018,
        3.26180286,
        0.13505771,
        6.32810228,
        3.30170823,
        3.1075047,
        3.3381902,
        6.20943094,
        3.50330669,
        3.25940823,
        3.13368048,
        3.27401205,
        0.1836304,
        3.26353402,
        4.63990877,
        0.20510497,
        3.16202071,
        3.21282786,
        0.0694859,
        6.19797663,
        0.10605501,
        3.23663232,
        3.07344717,
        0.11301269,
        6.13920848,
        3.3685336,
        3.09987176,
        3.04057022,
        6.4687988,
        3.09423741,
        6.36560324,
        3.23917211,
        1.02853817,
        3.35206863,
        6.07774304,
        3.24104718,
        0.83491118,
        6.29390083,
        3.19741791,
        6.24306857,
        3.17706214,
        3.3241074,
        6.31356853,
        3.13553537,
        0.21161276,
        3.23622737,
        6.30691142,
        3.1188802,
        6.3918742,
        0.50730425,
        6.22983498,
        0.13784034,
        3.30371674,
        6.35615831,
        3.24529634,
        3.12823302,
        0.03884129,
        6.17685171,
        3.27339587,
        6.25903412,
        0.07024611,
        3.22717155,
        6.23099972,
        0.14889149,
        3.1892271,
        6.07519259,
        0.14619932,
        3.29369482,
        3.22204459,
        3.29067816,
        3.15957082,
        6.20887812,
        2.11484691,
        6.34740603,
        3.19095153,
        6.06535949,
        3.08968063,
        3.20953152,
        3.18339847,
        3.31668822,
        0.18036691,
    ]

    def load_data():
        data = np.loadtxt("../data/predator-prey-data.csv", delimiter=",")
        return data[:, 0], data[:, 1], data[:, 2]

    raw_data = load_data()
    t_data = raw_data[0]
    x_data = raw_data[1]
    y_data = raw_data[2]

    import_x = x_data
    import_y = y_data

    global_ODE_param = [0.84321067, 0.426050229, 1.16945457, 2.03212881]
    global_SA_param = [
        np.array([1.0, 0.1, 0.1, 1.0]),  # initial_state
        [(0.1, 1), (0.1, 1), (0.1, 2), (0.1, 3)],  # bounds
        lambda x: x + np.random.normal(0, 0.1, size=len(x)),  # proposal_func
        100,  # initial_temp
        0.99,  # alpha
        800,  # max_iter, markov chain length
    ]

    print(f"Global ODE Parameters: {global_ODE_param}")
    #  print(f"Global Objective Value: {best_mse}")

<<<<<<< HEAD
    importance_x_removed, importance_y_removed = critical_number_check(
        x_data,
        y_data,
        t_data,
        import_x,
        import_y,
        fft_obj.objective_fft,
        global_SA_param,
        ic.global_objective,
        global_ODE_param,
        repeat=2,
    )

    # importance_x_removed = [5.81969154, 2.99311334, 3.02981609, 2.85487998, 0.419656, 0.42894168,
    #                         5.896001,   3.06497382, 0.44663044, 2.95670358, 0.46294484, 0.38066793,
    #                         5.97590395, 5.86687577, 2.94902943, 3.20738299 ,3.31381957, 3.13153569,
    #                         0.42170235, 2.99450703, 3.10861631, 3.18888738 ,0.36013955, 3.43399971,
    #                         5.84124929, 5.65289878, 3.41675953, 0.41391985, 3.05764565, 3.04165052,
    #                         3.46444158, 1.82138839, 3.21058334 ,0.40444788 ,0.32405545, 5.97609061,
    #                         3.23962544, 0.3360888 , 0.43740838, 3.05903027 ,3.21713256, 3.27867784,
    #                         3.29162417, 0.35980182, 0.41492452 ,0.56293174, 0.42811623, 3.18832976,
    #                         3.47126876, 5.8857237 , 3.10115583, 0.43264501 ,3.46479519, 3.46295479,
    #                         0.30001099, 6.00325819, 3.3699965  ,0.88091511, 0.28806255, 3.09874359,
    #                         3.15630354, 0.46304412, 5.96561269, 1.38081262, 5.77131416, 5.9881144,
    #                         0.32231207, 3.51953838,3.46161017 ,3.13587508, 0.39212755, 5.95673112,
    #                         3.14033853, 0.27719918, 6.7883381,  0.27457187, 4.12890841, 6.74538659,
    #                         2.99020532 ,3.1157776 , 6.55220051, 6.07908084, 0.29619197, 0.23405105,
    #                         7.35562752, 0.42907005, 0.22842169 ,0.4715245,  0.38403652, 4.19816558,
    #                         4.16880216, 0.17902304 ,3.40101575, 4.02094014, 7.65207074, 0.28652,
    #                         4.96371862, 8.2934929 , 6.04814504, 0.29121237]

    # importance_y_removed = [3.06769903,  0.37388657,  5.68866707,  0.36606939,  2.90858892,  2.98041644,
    #                         3.08912034 , 3.16658994,  5.75700688 , 0.34484724,  5.70869499 , 3.13491052,
    #                         0.37822129 , 0.45285739,  3.05035604,  5.63110762,  3.23075814 , 5.89080358,
    #                         5.7112456  , 0.4407757  , 5.49894897,  3.16422339,  3.09823345,  0.34940749,
    #                         2.98647575 , 0.41104439 , 3.17798285,  5.7052644 ,  0.44388708,  3.251678,
    #                         3.04471746,  5.81971309,  3.05169017,  3.09954753,  2.82117622 , 3.99006153,
    #                         5.56138737,  0.43634391,  3.00127972,  3.07036629,  0.4068173 ,  0.37159602,
    #                         3.05665965 , 0.40727148,  5.57929298,  5.51052272 , 5.76745926,  0.34599109,
    #                         5.76109648 , 0.49491005,  5.94036657,  3.08556638 , 2.83700979,  0.39742259,
    #                         0.41705986,  0.36645129,  0.39609007 , 0.41553713,  0.38772455,  2.86695133,
    #                         3.19103402,  3.03013217 , 0.35216175 , 0.35298911 , 3.15575241 , 0.48349097,
    #                         5.99876461,  0.35510841 , 0.41826395, 3.22744908,  3.33869166 , 0.3447972,
    #                         3.19166336,  3.18214168 , 0.32849298,  0.37322038 , 0.37019043 , 3.23185344,
    #                         0.3911022 ,  3.11699478 , 5.82005894 , 3.30787712 , 5.98845594 , 0.43118018,
    #                         0.35280056 , 0.31108592 , 5.92564682, 3.26718384 , 2.94252378 , 2.48621215,
    #                         3.2639738,   7.34295423 , 3.01207076 , 6.82396118,  3.98627865,  0.38257959,
    #                         7.933939 ,  15.27309687 , 0.71352007 , 3.15884646,]
=======
    importance_x_removed, importance_y_removed = critical_number_check(x_data, y_data, t_data, import_x, import_y, fft_obj.objective_fft, global_SA_param, ic.global_objective, global_ODE_param, repeat=2)
>>>>>>> refs/remotes/origin/main

    print(f"Importance of Removed X: {np.mean(importance_x_removed, axis=0)}")
    print(f"Importance of Removed Y: {np.mean(importance_y_removed, axis=0)}")

    plt.figure(figsize=(14, 8), dpi=300)
    # Plot importance of X
    plt.subplot(2, 1, 1)
    # plt.plot(np.mean(importance_x_removed, axis=0), label="Critcal points of X", marker='o', linestyle='--', color='b')
    plt.plot(
        np.mean(importance_x_removed, axis=0),
        label="Critcal points of X",
        marker="o",
        linestyle="--",
        color="b",
    )
    plt.title(
        "X Critical Points, Removing X by effect ascending order and Re-Opt the model, Calculate MSE in full data",
        fontsize=18,
    )
    plt.xlabel("Number of X removed", fontsize=20)
    plt.ylabel("MSE difference", fontsize=20)
    plt.legend(fontsize=12)

    # Plot importance of Y
    plt.subplot(2, 1, 2)
    # plt.plot(importance_y_removed, label="Critcal points of Y", marker='s', linestyle='--', color='r')
    plt.plot(
        np.mean(importance_y_removed, axis=0),
        label="Critcal points of Y",
        marker="s",
        linestyle="--",
        color="r",
    )
    plt.title(
        "Y Critical Points, Removing Y by effect ascending order and Re-Opt the model, Calculate MSE in full data",
        fontsize=18,
    )
    plt.xlabel("Number of Y removed", fontsize=20)
    plt.ylabel("MSE difference", fontsize=20)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
<<<<<<< HEAD


# Importance of Removed X: [5.81969154 2.99311334 3.02981609 2.85487998 0.419656   0.42894168
#  5.896001   3.06497382 0.44663044 2.95670358 0.46294484 0.38066793
#  5.97590395 5.86687577 2.94902943 3.20738299 3.31381957 3.13153569
#  0.42170235 2.99450703 3.10861631 3.18888738 0.36013955 3.43399971
#  5.84124929 5.65289878 3.41675953 0.41391985 3.05764565 3.04165052
#  3.46444158 1.82138839 3.21058334 0.40444788 0.32405545 5.97609061
#  3.23962544 0.3360888  0.43740838 3.05903027 3.21713256 3.27867784
#  3.29162417 0.35980182 0.41492452 0.56293174 0.42811623 3.18832976
#  3.47126876 5.8857237  3.10115583 0.43264501 3.46479519 3.46295479
#  0.30001099 6.00325819 3.3699965  0.88091511 0.28806255 3.09874359
#  3.15630354 0.46304412 5.96561269 1.38081262 5.77131416 5.9881144
#  0.32231207 3.51953838 3.46161017 3.13587508 0.39212755 5.95673112
#  3.14033853 0.27719918 6.7883381  0.27457187 4.12890841 6.74538659
#  2.99020532 3.1157776  6.55220051 6.07908084 0.29619197 0.23405105
#  7.35562752 0.42907005 0.22842169 0.4715245  0.38403652 4.19816558
#  4.16880216 0.17902304 3.40101575 4.02094014 7.65207074 0.28652
#  4.96371862 8.2934929  6.04814504 0.29121237]
# Importance of Removed Y: [ 3.06769903  0.37388657  5.68866707  0.36606939  2.90858892  2.98041644
#   3.08912034  3.16658994  5.75700688  0.34484724  5.70869499  3.13491052
#   0.37822129  0.45285739  3.05035604  5.63110762  3.23075814  5.89080358
#   5.7112456   0.4407757   5.49894897  3.16422339  3.09823345  0.34940749
#   2.98647575  0.41104439  3.17798285  5.7052644   0.44388708  3.251678
#   3.04471746  5.81971309  3.05169017  3.09954753  2.82117622  3.99006153
#   5.56138737  0.43634391  3.00127972  3.07036629  0.4068173   0.37159602
#   3.05665965  0.40727148  5.57929298  5.51052272  5.76745926  0.34599109
#   5.76109648  0.49491005  5.94036657  3.08556638  2.83700979  0.39742259
#   0.41705986  0.36645129  0.39609007  0.41553713  0.38772455  2.86695133
#   3.19103402  3.03013217  0.35216175  0.35298911  3.15575241  0.48349097
#   5.99876461  0.35510841  0.41826395  3.22744908  3.33869166  0.3447972
#   3.19166336  3.18214168  0.32849298  0.37322038  0.37019043  3.23185344
#   0.3911022   3.11699478  5.82005894  3.30787712  5.98845594  0.43118018
#   0.35280056  0.31108592  5.92564682  3.26718384  2.94252378  2.48621215
#   3.2639738   7.34295423  3.01207076  6.82396118  3.98627865  0.38257959
#   7.933939   15.27309687  0.71352007  3.15884646]

# Separate critical indices for X and Y
removed_indices_x = remove_points_around_indices(
    x_data, critical_indices_x, window_size
)
removed_indices_y = remove_points_around_indices(
    y_data, critical_indices_y, window_size
)
# Combine indices to modify all datasets consistently
combined_indices = sorted(set(removed_indices_x).union(set(removed_indices_y)))
x_data_modified = np.delete(x_data, combined_indices)
y_data_modified = np.delete(y_data, combined_indices)
t_data_modified = np.delete(t_data, combined_indices)
=======
>>>>>>> refs/remotes/origin/main
