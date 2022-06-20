from load_data_ex2 import *
from normalize_features import *
from gradient_descent import *
from calculate_hypothesis import *
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex2()

# Normalize
X_normalized, mean_vec, std_vec = normalize_features(X)

# After normalizing, we append a column of ones to X, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)
# print(X_normalized)

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((3))
alpha = 0.04
print("alpha is ", alpha)
iterations = 100

# plot predictions for every iteration?
do_plot = True

# call the gradient descent function to obtain the trained parameters theta_final
theta_final = gradient_descent(X_normalized, y, theta, alpha, iterations, do_plot)

#########################################
# Write your code here
# Create two new samples: (1650, 3) and (3000, 4)
# Calculate the hypothesis for each sample, using the trained parameters theta_final
# Make sure to apply the same preprocessing that was applied to the training data
# Print the predicted prices for the two samples

########################################/

test = np.array([[1650, 3], [3000, 4]])

# Normalize
# tn, _, _ = normalize_features(test)

tn = (test - mean_vec) / std_vec


# After normalizing, we append a column of ones to X, as the bias term
column_of_ones = np.ones((tn.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
tn1 = np.append(column_of_ones, tn, axis=1)
# tn2 = np.append(column_of_ones, tn2, axis=1)

print("theta_final is",theta_final)

h1 = calculate_hypothesis(tn1, theta_final, 0)
print("Prediction of [1650,3] is", h1)

h2 = calculate_hypothesis(tn1, theta_final, 1)
print("Prediction of [3000,1] is", h2)