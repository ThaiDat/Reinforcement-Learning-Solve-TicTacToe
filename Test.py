import pickle
from random import choice, seed
from DP import get_possible_moves, check_win_bit, gamma, theta, R, RLose

board = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]


def encode_board(board, side):
    '''Convert board to state'''
    if side != 1 and side != 2:
        return None

    # Encode each player moves as 9-bit number
    player1 = 0
    player2 = 0
    mask = 1
    for row in board:
        for c in row:
            if c == 1:
                player1 |= mask
            elif c == 2:
                player2 |= mask
            mask <<= 1

    return (player1, player2) if side == 1 else (player2, player1)


def decode_board(state):
    '''Convert state to board'''
    return [[1 if ((state[0] >> (row * 3 + i)) & 1) == 1 else
             2 if ((state[1] >> (row * 3 + i)) & 1) == 1 else 0
             for i in range(3)]
            for row in range(3)]


# Load value functions from file
Vfirst = None
with open('first.dic', 'rb') as f:
    Vfirst = pickle.load(f)

Vsecond = None
with open('second.dic', 'rb') as f:
    Vsecond = pickle.load(f)


def pick_movement(board, side):
    '''Pick a movement from a board'''
    if side != 1 and side != 2:
        return None
    state = encode_board(board, side)
    V = Vfirst if side == 1 else Vsecond

    # Find best move from all possible moves using corresponding value function
    moves = get_possible_moves(state)
    max_moves = []
    max_val = float('-inf')
    for move in moves:
        # Apply move
        action_temp_state = state[0] | (1 << move), state[1]
        if check_win_bit(action_temp_state[0]):
            return move

        # Find the value of current move using bellman optimal equation
        action_value = 0
        for environment_response in moves:
            if environment_response == move:
                continue
            next_state = action_temp_state[0], action_temp_state[1] | (1 << environment_response)
            reward = -1 if check_win_bit(next_state[1]) else 0
            action_value += (reward + gamma * V[next_state]) / (len(moves) - 1)

        # Work fine without theta
        # Using theta here for fair choice
        if abs(action_value - max_val) < theta:
            max_moves.append(move)
        elif action_value > max_val:
            max_val = action_value
            max_moves = [move]

    return None if len(max_moves) == 0 else choice(max_moves)


if __name__ == '__main__':
    seed(29031996)
    side = 1
    next_move = pick_movement(board, side)
    while next_move is not None:
        board[next_move // 3][next_move % 3] = side
        for row in board:
            print(row)
        print('--- --- ---')
        side = 3 - side
        next_move = pick_movement(board, side)
