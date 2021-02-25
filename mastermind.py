'''
TODO
'''

import time
import argparse
import itertools
from collections import namedtuple
from termcolor import colored

import numpy as np
from tqdm import tqdm
from scipy.stats import norm

parser = argparse.ArgumentParser('mastermind')
parser.add_argument('--num_simulations', '-n', dest='num_simulations', type=int, default=1)
parser.add_argument('--board-size', '-s', dest='board_size', type=int, default=4)
parser.add_argument('--num-colors', '-c', dest='num_colors', type=int, default=7)

parser.add_argument('--display', '-d', dest='display', action='store_true')
parser.set_defaults(display=False)

args = parser.parse_args()

num_simulations = args.num_simulations
board_size = args.board_size
num_colors = args.num_colors
num_colors = args.num_colors
display = args.display

FinishedGame = namedtuple('FinishedGame', 'trials speed')
board = []

def generate_random_trial() -> np.ndarray:
    '''
    TODO
    '''
    return np.random.randint(low=1, high=num_colors, size=board_size)

def count_per_color(trial: np.ndarray) -> np.ndarray:
    '''
    TODO
    '''
    return np.bincount(trial, minlength=num_colors+1)

# Computing hints for the current row
def compute_hints(trial: np.ndarray, solution: np.ndarray) -> dict:
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

def process_trial(trial: np.ndarray, solution: np.ndarray) -> dict[str, np.ndarray]:
    '''
    TODO
    '''
    line = {
        'trial': trial,
        'hints': compute_hints(trial, solution)
    }

    return line

def update_possibilities(possibilities: np.ndarray) -> np.ndarray:
    
    trial = board[-1]['trial']
    hints = board[-1]['hints']

    new_possibilities = np.array([possibility for possibility in possibilities if hints == compute_hints(trial, possibility)])

    return new_possibilities

def generate_random_trial_from_possibility(possibilities: np.ndarray) -> np.ndarray:
    '''
    TODO
    '''

    random_index = np.random.choice(possibilities.shape[0], replace=False)
    return possibilities[random_index]

def single_game() -> FinishedGame:
    '''
    TODO
    '''
    global board

    start_simluation = time.time()

    board = []
    solution = generate_random_trial()
    possibilities = np.array(list(itertools.product(range(1, num_colors+1), repeat=board_size)))
    while True:
        trial = generate_random_trial_from_possibility(possibilities)
        
        board.append(process_trial(trial, solution))
        board[-1]['possibilities'] = len(possibilities)

        possibilities = update_possibilities(possibilities)
        if board[-1]['hints']['correct_color_position'] == board_size:
            break
    
    if display == True:
        print_colored_board(board, solution)

    return FinishedGame(trials=len(board), speed=time.time() - start_simluation)

pegs = ['█', '░', '▓']
colors = ['grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
colored_pegs = list(itertools.product(pegs, colors))
if len(colored_pegs) < num_colors:
    raise ValueError('Number of colors too high for console print.')

def print_colored_row(pegs: np.ndarray, hints: np.ndarray=None) -> None:
    print(''.join(colored(*(colored_pegs[peg])) for peg in pegs), end=' ')
    if hints is not None:
        ccp = hints['correct_color_position']
        cc = hints['correct_color']
        print('●'*ccp + '○'*cc + ' '*(board_size-cc-ccp) , end=' ')

def print_colored_board(board: list[dict[str, np.ndarray]], solution: np.ndarray) -> None:    
    print('###########')
    for i, row in enumerate(board):
        print(i+1, end=' ')
        print_colored_row(row['trial'], row['hints'])
        print(f"{row['possibilities']}", end='')
        print('')
    print('-----------')
    print('S', end=' ')
    print_colored_row(solution)
    print('')

def print_game_stats(games: list[FinishedGame]) -> None:
    '''
    TODO
    '''
    mean_length, var_length = norm.fit([game.trials for game in games])
    mean_time, var_time = norm.fit([game.speed for game in games])

    print(f'Average length: {mean_length:.2}')
    print(f'Variance length: {var_length:.2}')
    print(f'Average time: {mean_time:.2}')
    print(f'Variance time: {var_time:.2}')


if __name__ == '__main__':
    start_time = time.time()

    finished_games = [single_game() for _ in tqdm(range(num_simulations), desc = 'Processing simulations')]

    print(f'Board of size {board_size} which {num_colors} different colors')
    print_game_stats(finished_games)
    print(f'--- {time.time() - start_time:.2} seconds ---')
