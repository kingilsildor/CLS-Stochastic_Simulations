import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Lotka-Volterra模型定义
def predator_prey_model(z, t, alpha, beta, delta, gamma):
    prey, predator = z
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    return [dprey_dt, dpredator_dt]

# 模拟退火算法定义
def simulated_annealing(data, t, initial_params, bounds, max_iter, cooling_rate):
    def objective_function(params):
        alpha, beta, delta, gamma = params
        sol = odeint(predator_prey_model, [data[0, 1], data[0, 2]], t, args=(alpha, beta, delta, gamma))
        error = np.sqrt(np.mean((data[:, 1] - sol[:, 0])**2 + (data[:, 2] - sol[:, 1])**2))
        return error

    current_params = np.array(initial_params)
    current_cost = objective_function(current_params)
    best_params = current_params.copy()
    best_cost = current_cost

    for i in range(max_iter):
        temp = max_iter / (i + 1) * cooling_rate
        new_params = current_params + np.random.uniform(-1, 1, size=4) * temp
        new_params = np.clip(new_params, [b[0] for b in bounds], [b[1] for b in bounds])
        new_cost = objective_function(new_params)

        if new_cost < current_cost or np.random.rand() < np.exp(-(new_cost - current_cost) / temp):
            current_params = new_params
            current_cost = new_cost

            if new_cost < best_cost:
                best_params = new_params
                best_cost = new_cost

        print(f"Iteration {i+1}: Best Cost = {best_cost:.5f}")

    return best_params

# 加载数据
def load_data():
    data = np.loadtxt("../observed_data/predator-prey-data.csv", delimiter=",")
    return data

data = load_data()
t = data[:, 0]

# 改进后的初始参数和退火算法超参数
initial_params = [0.84321067, 0.426050229, 1.16945457, 2.03212881] # 更贴近真实值的初始参数
bounds = [(0, 1), (0, 0.05), (0, 0.05), (0, 1)]
max_iter = 1000  # 增加最大迭代次数
cooling_rate = 0.95  # 减慢冷却速率

# 执行模拟退火算法
estimated_params = simulated_annealing(data, t, initial_params, bounds, max_iter, cooling_rate)
print("Estimated Parameters:", estimated_params)

# 使用估计参数模拟结果
sol_estimated = odeint(predator_prey_model, [data[0, 1], data[0, 2]], t, args=tuple(estimated_params))

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(t, data[:, 1], 'o', label='Prey (Data)', alpha=0.6)
plt.plot(t, data[:, 2], 'o', label='Predator (Data)', alpha=0.6)
plt.plot(t, sol_estimated[:, 0], '--', label='Prey (Estimated)', alpha=0.8)
plt.plot(t, sol_estimated[:, 1], '--', label='Predator (Estimated)', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('Simulated Annealing for Lotka-Volterra Model')
plt.grid()
plt.show()
