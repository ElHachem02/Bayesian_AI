"""Solution."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel, DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor



# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
MAX_ITERATIONS = 50   # Define the maximum number of iterations for Bayesian optimization
STD_BIOAVAILABLE = 0.15
STD_SA = 1e-4

# Implement a self-contained solution in the BO_algo class.
# main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.

        # Initialize the Gaussian Process for bioavailability (logP)
        kernel_f = 1.32**2 * Matern(length_scale=3.25, nu=2.5) + WhiteKernel(noise_level=0.482)
        self.gp_f = GaussianProcessRegressor(kernel=kernel_f, n_restarts_optimizer=5, normalize_y=True, random_state=42)

        # Initialize the Gaussian Process for synthesizability (SA)
        kernel_v = 0.00316**2 * DotProduct(sigma_0=0.000248) + 0.309**2 * Matern(length_scale=0.393, nu=2.5) + WhiteKernel(noise_level=0.909)
        self.gp_v = GaussianProcessRegressor(kernel=kernel_v, n_restarts_optimizer=5, normalize_y=True, random_state=42)

        #Define lambda penalty
        self.lambda_penalty = 2  # Penalty weight for constraint violation

        # Initialize data storage for observations
        self.sampledPoints = np.empty((0, 1))
        self.f_values = np.array([])
        self.v_values = np.array([])

        self.beta = 1.96
        

    def meets_constraint(self, x):
        predicted_sa, std = self.gp_v.predict(x, return_std=True)

        return predicted_sa + std * self.beta < SAFETY_THRESHOLD


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        max_tries = 10 
        lambda_increase_factor = 1.5

        smallest_v = float('inf')
        best_x = None

        for _ in range(max_tries):
            x_next = np.array([[self.optimize_acquisition_function()]])
            v_mean, v_std = self.gp_v.predict(x_next, return_std=True)
            v_value = v_mean + self.beta * v_std

            if v_value < smallest_v:
                smallest_v = v_value
                best_x = x_next

            if self.meets_constraint(x_next):
                return x_next
            else:
                self.lambda_penalty *= lambda_increase_factor  # Increase lambda_penalt
        
        return best_x
        
        
    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt
    

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # Implement the acquisition function you want to optimize.
        meanPrediction, stdDeviation = self.gp_f.predict(x, return_std=True)
        meanSa, stdSa = self.gp_v.predict(x, return_std=True)

        penalty = np.maximum(meanSa + stdSa * self.beta - SAFETY_THRESHOLD, 0)

        ucb = meanPrediction + 1.08 * stdDeviation
        return ucb - self.lambda_penalty*penalty

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.

        self.sampledPoints = np.vstack([self.sampledPoints, x])
        self.f_values = np.append(self.f_values, f)
        self.v_values = np.append(self.v_values, v)

        self.gp_f.fit(self.sampledPoints, self.f_values)
        self.gp_v.fit(self.sampledPoints, self.v_values)


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        '''
        best_solution = None
        best_value = -np.inf
        nonConstraintMeetingPoints = []

        for _ in range(MAX_ITERATIONS):
            x_next = self.next_recommendation()
            y_f = self.gp_f.predict(x_next)
            y_v = self.gp_v.predict(x_next)

            self.add_data_point(x_next, y_f, y_v)

            if self.meets_constraint(x_next):
                if y_f > best_value:
                    best_value = y_f
                    best_solution = x_next
            else:
                nonConstraintMeetingPoints.append((x_next, y_v))

        if(best_solution==None):
            best_solution = sorted(nonConstraintMeetingPoints, key=lambda x: x[1])[0].first

        return self.best_solution

        '''

        bestF = self.f_values[-1]
        bestSolution = self.sampledPoints[-1]
        for i in range(self.sampledPoints.shape[0]):
            if self.meets_constraint([[self.v_values[i]]]):
                if self.f_values[i] > bestF:
                    bestSolution = self.sampledPoints[i]

        return bestSolution
    
    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        return
        # Generating a sequence of points within the domain
        x = np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], 100).reshape(-1, 1)

        # Predicting for objective function
        mean_f, std_f = self.gp_f.predict(x, return_std=True)

        # Predicting for constraint function
        mean_v, std_v = self.gp_v.predict(x, return_std=True)

        # Plotting the objective function
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, mean_f, 'r', lw=2)
        plt.fill_between(x.ravel(), mean_f - 1.96 * std_f, mean_f + 1.96 * std_f, alpha=0.5)
        plt.title("Objective Function (f)")
        plt.xlabel("x")
        plt.ylabel("f(x)")

        # Plotting the constraint function
        plt.subplot(1, 2, 2)
        plt.plot(x, mean_v, 'b', lw=2)
        plt.fill_between(x.ravel(), mean_v - 1.96 * std_v, mean_v + 1.96 * std_v, alpha=0.5)
        plt.axhline(SAFETY_THRESHOLD, color='k', linestyle='--')
        plt.title("Constraint Function (v)")
        plt.xlabel("x")
        plt.ylabel("v(x)")

        # Optionally plotting the recommended point
        if plot_recommendation:
            recommended_x = self.next_recommendation()
            recommended_f, _ = self.gp_f.predict(recommended_x, return_std=True)
            recommended_v, _ = self.gp_v.predict(recommended_x, return_std=True)

            plt.subplot(1, 2, 1)
            plt.scatter(recommended_x, recommended_f, c='green', marker='*', s=100, label='Recommended Point')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.scatter(recommended_x, recommended_v, c='green', marker='*', s=100, label='Recommended Point')
            plt.legend()

        plt.tight_layout()
        plt.show()


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Initialize unsafe evaluations counter
    unsafe_evals_count = 0

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()
        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)
        agent.plot()

        # Check if the evaluation is unsafe
        if not agent.meets_constraint(x):
            unsafe_evals_count += 1


    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals {unsafe_evals_count}\n')

    return agent

if __name__ == "__main__":
    main()