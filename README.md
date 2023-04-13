# GridWorld Q-Network Agent

This repository contains code for a Q-network agent trained to learn the optimal policy for a GridWorld environment. The agent uses a neural network to estimate the Q-values for each state-action pair and is trained using the Q-learning algorithm.

## Requirements

Python 3.x
PyTorch
Matplotlib
Numpy

## Usage

To run the code, simply execute the gridworld_q_network.ipynb file. 

## The file contains the following code:

The GridWorld class, which defines a GridWorld environment with a specified size and goal and obstacle positions.
The QNetwork class, which defines a neural network with a specified number of input and output dimensions.
The train_agent function, which trains the Q-network agent to learn the optimal policy for the GridWorld environment using the Q-learning algorithm.
The test_agent function, which tests the trained agent on the GridWorld environment to find the optimal path.
Code to initialize a GridWorld environment, create a Q-network, train the agent using the train_agent function, test the agent using the test_agent function, and plot the results.
You can modify the hyperparameters of the agent, such as the learning rate, discount factor, and exploration rate, in the code to experiment with different settings.

## Results

The code outputs the following results:

The initial state of the GridWorld environment.
The final state of the GridWorld environment with the optimal path found by the agent.
The Q-values table, which shows the Q-values for each state-action pair learned by the agent.
The number of steps taken during training, which shows the convergence of the agent's performance over time.
The MSE convergence, which shows the convergence of the agent's Q-value estimates over time.
The trajectory of each weight during training, which shows how the neural network weights change over time.


## License

This code is licensed under the MIT License. See the LICENSE file for more information.