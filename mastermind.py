'''
TODO
'''

import time
import argparse
from collections import namedtuple

import numpy as np
from tqdm import tqdm
from scipy.stats import norm

parser = argparse.ArgumentParser("mastermind")
parser.add_argument("--num_simulations", "-n", dest="num_simulations", type=int, default=1)
parser.add_argument("--board-size", "-s", dest="board_size", type=int, default=4)
parser.add_argument("--num-colors", "-c", dest="num_colors", type=int, default=7)

args = parser.parse_args()

num_simulations = args.num_simulations
board_size = args.board_size
num_colors = args.num_colors

FinishedGame = namedtuple('FinishedGame', 'trials, speed')
board = []

def generate_random_trial():
    '''
    TODO
    '''
    return np.random.randint(low=1, high=num_colors, size=board_size)

def generate_empty_trial():
    '''
    TODO
    '''
    return np.zeros(shape=board_size, dtype=int)

def count_per_color(trial):
    '''
    TODO
    '''
    return np.bincount(trial, minlength=num_colors)

# Generation of a first try
# In the case of a game of : size = 4, color = 7, it will generate the following board : [1,2,3,4]
# This is used to maximise the information from the first try
def generate_first_trial():
    '''
    TODO
    '''
    return np.minimum(np.arange(start=1, stop=board_size + 1), np.full(shape=board_size, fill_value=num_colors))

# Maximizing the second try based on the information of the first one
def generate_second_trial():
    '''
    TODO
    '''
    last_trial = board[-1]['trial']
    last_hints = board[-1]['hints']
    cor_col = last_hints['correct_color']
    cor_col_pos = last_hints['correct_color_position']

    next_trial = generate_empty_trial()
    next_trial[:cor_col_pos] = last_trial[:cor_col_pos]
    for pos in range(cor_col_pos, cor_col_pos + cor_col):
        if pos+1 < board_size:
            next_trial[pos+1] = last_trial[pos]
        else:
            next_trial[cor_col_pos] = last_trial[pos]

    for i, keg in enumerate(next_trial):
        if not keg:
            if board_size < num_colors:
                next_trial[i] = np.random.randint(low=board_size+1, high=num_colors)
            else:
                next_trial[i] = np.random.randint(low=1, high=num_colors-1)

    return next_trial

# Computing hints for the current row
def compute_hints(trial, solution):
    '''
    TODO
    '''

    hints = {
        'correct_color': 0,
        'correct_color_position': 0,
    }

    hints['correct_color'] = np.minimum(count_per_color(trial), count_per_color(solution)).sum()

    for peg_trial, peg_solution in zip(trial, solution):
        if peg_trial == peg_solution:
            hints['correct_color'] -= 1
            hints['correct_color_position'] += 1

    return hints

def validate_trial(trial):
    '''
    TODO
    '''

    for line in reversed(board):
        past_trial = line['trial']
        past_hints = line['hints']

        hypotetical_hints = compute_hints(trial, past_trial)
        if past_hints != hypotetical_hints:
            return False

    return True

# After the first two trys, this will generate a row that verifies all the previous equations
def generate_trial():
    '''
    TODO
    '''

    yield generate_first_trial()
    yield generate_second_trial()
    while True:
        trial = generate_random_trial()
        if validate_trial(trial):
            yield trial

def process_trial(trial, solution):
    '''
    TODO
    '''
    line = {
        'trial': trial,
        'hints': compute_hints(trial, solution)
    }

    return line

def single_game():
    '''
    TODO
    '''
    global board
    start_simluation = time.time()

    solution = generate_random_trial()
    board = []
    for trial in generate_trial():
        board.append(process_trial(trial, solution))
        if board[-1]['hints']['correct_color_position'] == board_size:
            break

    return FinishedGame(trials=len(board), speed=time.time() - start_simluation)

def print_game_stats(games):
    '''
    TODO
    '''
    mean_length, var_length = norm.fit([game[0] for game in games])
    mean_time, var_time = norm.fit([game[1] for game in games])

    print(f'Average length: {mean_length}')
    print(f'Variance length: {var_length}')
    print(f'Average time: {mean_time}')
    print(f'Variance time: {var_time}')

start_time = time.time()

finished_games = [single_game() for _ in tqdm(range(num_simulations), desc = 'Processing simulations')]

print(f'Board of size {board_size} which {num_colors} different colors')
print_game_stats(finished_games)
print('--- {time.time() - start_time} seconds ---')
