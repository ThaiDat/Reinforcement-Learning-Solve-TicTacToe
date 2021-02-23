import pickle

# A state consist of two 9-bit numbers.
# Each number is the position of all movements each player made
# The first number is for current player.
# state = (0, 0)

# A dictionary hold board state as key and an array of probabilty as value
# policy = dict()

# Value function is a dictionary hold board state as key and a number as value
# value function for first player - player1
first_value_function = dict()

# value function for second player - player2
second_value_function = dict()

# All cells of a board
all_cells = list(range(9))

# Discount parameter
gamma = 0.9

# Changed control parameter
theta = 0.001


# Reward
R = 0
RLose = -1
RWin = 1


def check_bit(num, pos):
    '''Check if the bit as pos is not empty'''
    return ((num >> pos) & 1) == 1


def check_win_bit(num):
    '''Check if the number holding state is a win state'''
    if check_bit(num, 0):
        if (check_bit(num, 1) and check_bit(num, 2)) or\
                (check_bit(num, 3) and check_bit(num, 6)) or\
                (check_bit(num, 4) and check_bit(num, 8)):
            return True
    if check_bit(num, 2):
        if (check_bit(num, 4) and check_bit(num, 6)) or\
                (check_bit(num, 5) and check_bit(num, 8)):
            return True
    if check_bit(num, 4):
        if (check_bit(num, 1) and check_bit(num, 7)) or\
                (check_bit(num, 3) and check_bit(num, 5)):
            return True
    if check_bit(num, 6) and check_bit(num, 7) and check_bit(num, 8):
        return True
    return False


def enable_bit(num, pos):
    '''Enable a bit at pos'''
    return num | (1 << pos)


def get_possible_moves(state):
    '''Get a list of empty cells from state'''
    return [cell for cell in all_cells
            if not (check_bit(state[0], cell) or check_bit(state[1], cell))]


def value_iteration(V):
    '''Improve value function model'''
    while True:
        delta = 0
        for state in V:
            oldval = V[state]

            # terminal state
            if check_win_bit(state[0]):
                V[state] = RWin + gamma * V[state]
                continue
            if check_win_bit(state[1]):
                V[state] = RLose + gamma * V[state]
                continue
            possible_moves = get_possible_moves(state)
            if len(possible_moves) == 0:
                V[state] = R + gamma * V[state]
                continue

            # Bellman optimality update
            maxval = float('-inf')
            for move in possible_moves:
                action_temp_state = enable_bit(state[0], move), state[1]
                reward = 0
                stateval = 0

                # Check if move cause end game
                if check_win_bit(action_temp_state[0]):
                    reward = 1
                    stateval = reward + gamma * V[action_temp_state]
                    if stateval > maxval:
                        maxval = stateval
                    continue

                # Environment response to action.
                for other_move in possible_moves:
                    if other_move == move:
                        continue

                    next_state = action_temp_state[0], enable_bit(action_temp_state[1], other_move)
                    if check_win_bit(next_state[1]):
                        reward = RLose
                    else:
                        reward = R
                    stateval += (reward + gamma * V[next_state]) / (len(possible_moves) - 1)

                # maximize state value
                if stateval > maxval:
                    maxval = stateval

            V[state] = maxval

            # delta control if value function is nearly equal to optimality value function
            delta = max(delta, abs(oldval - maxval))

        # Terminating condition
        if delta < theta:
            break


def init_states():
    '''Init all states in policy and value function'''
    def recursive_init(state):
        '''
        Init states recursively.
        On each state, We will init 2 moves, one for each player.
        '''
        moves1, moves2 = state

        possible_moves = get_possible_moves(state)
        # Player1 make move
        for move1 in possible_moves:
            # State for player1 is (moves of player1, moves of player2)
            # state for player2 is (moves of player2, moves of player1)
            changed_moves1 = enable_bit(moves1, move1)
            if (moves2, changed_moves1) in second_value_function:
                continue
            second_value_function[(moves2, changed_moves1)] = 0
            # Check if player1 won
            if check_win_bit(changed_moves1):
                first_value_function[(changed_moves1, moves2)] = 0
                continue

            # Player2 move
            for move2 in possible_moves:
                if move2 == move1:
                    continue

                changed_moves2 = enable_bit(moves2, move2)
                if (changed_moves1, changed_moves2) in first_value_function:
                    continue
                first_value_function[(changed_moves1, changed_moves2)] = 0
                # Check if player2 won
                if check_win_bit(changed_moves2):
                    second_value_function[(changed_moves2, changed_moves1)] = 0
                    continue
                recursive_init((changed_moves1, changed_moves2))
    # Init all states of each value function with zero value
    first_value_function[(0, 0)] = 0
    recursive_init((0, 0))


if __name__ == '__main__':
    init_states()
    value_iteration(first_value_function)
    value_iteration(second_value_function)

    # Save value function to file
    with open('first.dic', 'wb') as f:
        pickle.dump(first_value_function, f)
    with open('second.dic', 'wb') as f:
        pickle.dump(second_value_function, f)
