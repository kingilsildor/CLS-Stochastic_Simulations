import numpy as np

def simulated_annealing(
    objective_h,
    true_data,
    initial_state,
    bounds,
    proposal_func,
    initial_temp,
    alpha,
    max_iter,
):
    """
    Simulated Annealing Algorithm using Metropolis-Hastings sampling.

    Parameters:
    ------------
    - objective_h: callable, the objective function `h(x)` to minimize.
    - initial_state: array-like, the initial state `x0`.
    - proposal_func: callable, generates a new state `x_i+1'` given current state `x_i`, e.g., normal distribution, N(x_i, sigma).
    - initial_temp: float, the initial temperature `T0`.
    - alpha: float, cooling rate (e.g., 0.9).
    - max_iter: int, maximum number of iterations.
    - noise: bool, whether to add noise to the data.

    Returns:
    ------------
    - best_state: array-like, the best state found.
    - best_value: float, the value of the objective function at the best state.
    """
    # Initialize
    x_current_state = initial_state
    h_current_value = objective_h(x_current_state, *true_data)
    best_state = x_current_state
    best_value = h_current_value
    temperature = initial_temp

    for i in range(max_iter):
        # TODO: record each iteration's the h_new_value and x_new_state.

        # Step 1: Generate a new candidate state using proposal function
        x_new_state = proposal_func(x_current_state)
        x_new_state = np.clip(
            x_new_state, *zip(*bounds)
        )  # Ensure parameters stay within bounds
        h_new_value = objective_h(x_new_state, *true_data)

        # Step 2: Compute the acceptance probability
        delta = h_new_value - h_current_value
        acceptance_prob = min(1, np.exp(-delta * temperature))

        # Step 3: Decide whether to accept the new state
        if delta < 0 or np.random.rand() < acceptance_prob:
            x_current_state = x_new_state
            h_current_value = h_new_value

            # Update the best state found so far
            if h_current_value < best_value:
                best_state = x_current_state
                best_value = h_current_value

        # Step 4: Decrease the temperature
        temperature *= alpha

    return best_state, best_value

def hill_climbing(objective, proposal_func, true_data, init_params, max_iter, tol):
    """
    Hill Climbing Algorithm.

    Parameters:
    ------------
    - objective: callable, the objective function to minimize.
    - proposal_func: callable, generates a new state given the current state.
    - init_params: array-like, the initial state.
    - max_iter: int, maximum number of iterations.
    - tol: float, tolerance for stopping criterion.
    - noise: bool, whether to add noise to the data.

    Returns:
    ------------
    - best_params: array-like, the best state found.
    - best_error: float, the value of the objective function at the best state.
    """
    current_params = init_params
    best_params = current_params
    best_error = objective(current_params, *true_data)

    for i in range(max_iter):
        new_params = proposal_func(current_params)
        new_error = objective(new_params, *true_data)

        if new_error < best_error:
            best_error = new_error
            best_params = new_params
            current_params = new_params
        
        if np.abs(best_error - new_error) < tol:
            break

    return best_params, best_error
