from calculate_hypothesis import *
from compute_cost import *
from plot_cost_train_test import *

def gradient_descent_training(X_train, y_train, X_test, y_test, theta, alpha, iterations):
    """
        :param X_train      : 2D array of our training set
        :param y_train      : 1D array of the groundtruth labels of the training set
        :param X_test       : 2D array of our test set
        :param y_test       : 1D array of the groundtruth labels of the test set
        :param theta        : 1D array of the trainable parameters
        :param alpha        : scalar, learning rate
        :param iterations   : scalar, number of gradient descent iterations
    """
    
    m = X_train.shape[0] # the number of training samples is the number of rows of array X
    cost_vector_train = np.array([], dtype=np.float32) # empty array to store the train cost for every iteration
    cost_vector_test = np.array([], dtype=np.float32) # empty array to store the test cost for every iteration

    # Gradient Descent
    for it in range(iterations):
        
        # initialize temporary theta, as a copy of the existing theta array
        theta_temp = theta.copy()
        sigma = np.zeros((len(theta)))

        for j in range(len(theta)):
            for i in range(m):
                #########################################
                # Write your code here
                # Calculate the hypothesis for the i-th sample of X, with a call to the "calculate_hypothesis" function

                ########################################/
                output1 = y_train[i]
                hypothesis = calculate_hypothesis(X_train, theta_temp, i)

                #########################################
                # Write your code here
                # Adapt the code, to compute the values of sigma for all the elements of theta

                ########################################/
                sigma[j] += (hypothesis - output1) * X_train[i, j]

            # update theta_temp
            theta_temp[j] = theta_temp[j] - alpha * sigma[j]

        ###############
        # copy theta_temp to theta
        theta = theta_temp.copy()
        
        # append current iteration's cost to cost vector
        #########################################
        # Write your code here
        # Store costs for both train and test set in their corresponding vectors
        
        ########################################/
        it_cost_vector_train = compute_cost(X_train, y_train, theta)
        it_cost_vector_test = compute_cost(X_test, y_test, theta)

        cost_vector_train = np.append(cost_vector_train, it_cost_vector_train)
        cost_vector_test = np.append(cost_vector_test, it_cost_vector_test)


    print('Gradient descent finished.')
    
    return theta, cost_vector_train, cost_vector_test