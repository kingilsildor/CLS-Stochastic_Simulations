# Predator-Prey System

This repository contains an implementation of a Predator-Prey System using the Lotka-Volterra equations. It includes various modules for modeling, analyzing, and visualizing the dynamics of predator-prey interactions. The system also integrates optimization algorithms like simulated annealing to fit models to observed data.

---

## Features

- **Lotka-Volterra Model:** Simulates predator-prey dynamics with customizable parameters.
- **Simulated Annealing Optimization:** Finds optimal parameters for the system by minimizing mean squared error (MSE).
- **Sensitivity Analysis:** Evaluates the impact of removing data points on model performance.
- **Weighted Mean Squared Error (MSE):** Assigns different weights to data points for robust fitting.
- **Fourier Analysis:** Compares observed and simulated data in the frequency domain.
- **Visualization Tools:** Plots for simulated vs. observed data and model performance.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HarvestStars/Predator-Prey-System.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Predator-Prey-System
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Run the Simulation
To run the Lotka-Volterra simulation:
```bash
python src/simulation.py
```

### 2. Perform Optimization
To optimize model parameters using simulated annealing:
```bash
python src/optimization.py
```

### 3. Conduct Sensitivity Analysis
To evaluate the importance of data points:
```bash
python src/sensitivity_analysis.py
```

### 4. Fourier Analysis
To analyze frequency-domain similarity between observed and simulated data:
```bash
python src/fourier_analysis.py
```

### 5. Visualization
To generate plots of the predator-prey dynamics:
```bash
python src/visualization.py
```

---

## Repository Structure

```
Predator-Prey-System/
├── src/
│   ├── lotka_volterra.py      # Implements Lotka-Volterra equations
│   ├── simulation.py          # Runs the simulation
│   ├── optimization.py        # Simulated annealing optimization
│   ├── sensitivity_analysis.py# Data sensitivity analysis
│   ├── fourier_analysis.py    # Fourier analysis of data
│   ├── visualization.py       # Visualization tools
│   ├── weighted_objective.py  # Weighted MSE objective function
├── data/
│   ├── observed_data.csv      # Example dataset
│   └── simulated_data.csv     # Generated simulation data
├── tests/
│   ├── test_simulation.py     # Unit tests for simulation
│   ├── test_optimization.py   # Unit tests for optimization
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── LICENSE                    # License file
```

---

## Key Equations

### Lotka-Volterra Equations
The predator-prey interactions are modeled using the following equations:
\[
\begin{aligned}
\frac{dx}{dt} &= \alpha x - \beta x y \\
\frac{dy}{dt} &= \delta x y - \gamma y
\end{aligned}
\]

### Weighted Mean Squared Error (MSE)
\[
\text{Weighted MSE} = \frac{1}{N} \sum_{i=1}^{N} \left[ w_x^{(i)} \cdot (x_{\text{sim}}^{(i)} - x_{\text{data}}^{(i)})^2 + w_y^{(i)} \cdot (y_{\text{sim}}^{(i)} - y_{\text{data}}^{(i)})^2 \right]
\]

---

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact
For any questions or suggestions, feel free to open an issue or contact the repository owner via GitHub.