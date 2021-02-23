# Reinforcement-Learning-Solve-TicTacToe

A TicTacToe solver based on Reinforcement Learning, using Dynamic Programming approach.

A solver means it completely solve the game, provide action and valuation for every board state.

# Design


A player will have a set of states. Each state is the board that the player has to make a move. After making a move, the environment (or the other player) will respond and transit to another state (that current player has to make a move)
More clearly, we have a state, we make an action, the environment responds and we come to another state.

To improve the performance of the agent over time, we use a value iteration and policy improvement theorem to update the value function of each state based on Bellman's optimality equation.

# Run and Test
Run `DynamicProgramming.py` to start evaluating the TicTacToe, it will result in 2 files `first.dic` and `second.dic`, they are dictionaries of board state (key) and value function (value) for each player.
Run `Test.py` to test. It will load 2 above files and simulate a game with 2 players. Change the seed will change the way each player play the game. This file also shows how to use the above pre-evaluated files.
