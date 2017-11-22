import numpy.random as random
import time

def generate_random(size, nbcolors):
	return [int((nbcolors - 1) * random.random() + 1) for i in xrange(size)]

def generate_one_each(size, nbcolors):
	essai = []
	for i in xrange(size):
		if i < nbcolors:
			essai.append(i+1)
		else:
			essai.append(nbcolors)
	return essai

def first_try(size, nbcolors, solution):
	essai1 = generate_one_each(size, nbcolors)
	indices = generate_indices(size, nbcolors, essai1, solution)
	
	return [[essai1, indices]];

def second_try(size, nbcolors, board, solution):
	indices1 = board[0][1] 
	essai1 = board[0][0]
	essai2 = [-1] * size
	noirs = indices1.count(2)
	blancs = indices1.count(1)

	for i in xrange(noirs):
		essai2[i] = essai1[i]

	for i in xrange(noirs, noirs + blancs):
		if i + 1 > (size - 1):
			next_indice = noirs
		else:
			next_indice = i + 1
		essai2[next_indice] = essai1[i]

	for a in essai2:
		if a == -1:
			if size < nbcolors:
				essai2[essai2.index(a)] = int((nbcolors - size) * random.random() + size + 1)
			else:
				essai2[essai2.index(a)] = int((nbcolors - 1) * random.random() + 1)

	indices2 = generate_indices(size, nbcolors, essai2, solution)

	return [essai2, indices2];

def generate_indices(size, nbcolors, essai, solution):
	indices = []
	nums_essai = [0] * nbcolors
	nums_solution = [0] * nbcolors
	
	for a in essai: 
		nums_essai[a - 1] += 1

	for a in solution: 
		nums_solution[a - 1] += 1

	for a,b in zip(nums_essai, nums_solution):
		for x in xrange(min(a,b)):
			indices.append(1)

	for a,b in zip(essai, solution):
		if a == b:
			indices[indices.index(1)] = 2
			
	return indices;

def essai_valid(size, nbcolors, board, essai, tested):

	nums_essai = [0] * nbcolors
	for a in essai:
		nums_essai[a - 1] += 1

	for ligne in board:
		nums_past = [0] * nbcolors
		indices = []

		for a in ligne[0]:
			nums_past[a - 1] += 1

		for a,b in zip(nums_essai, nums_past):
			for x in xrange(min(a,b)):
				indices.append(1)	

		if len(indices) != len(ligne[1]):
			return False; 

		for past_pion, futur_pion in zip(ligne[0], essai):
			if past_pion == futur_pion:
				indices[indices.index(1)] = 2
		
		if indices.count(2) != ligne[1].count(2):
			return False;

	return True;	

def find_answer(size, nbcolors, board, solution, tested):
	essaiX = generate_random(size, nbcolors)

	while essai_valid(size, nbcolors, board, essaiX, tested) != True:
		tested.append(essaiX)
		essaiX = generate_random(size, nbcolors)
	
	indicesX = generate_indices(size, nbcolors, essaiX, solution)
	
	return [essaiX, indicesX];

def single_game(size, nbcolors):
	start_simluation = time.time()
	solution = generate_random(size, nbcolors)
	board = first_try(size, nbcolors, solution)
	tested = []

	board.append(second_try(size, nbcolors, board, solution))

	while board[-1][1] != [2] * size:
		board.append(find_answer(size, nbcolors, board, solution, tested))
	
	return [len(board), time.time() - start_simluation];

def sum(dataset):
	dataset_sum = 0

	for d in dataset:
		dataset_sum += d;

	return dataset_sum

def average(dataset):
	return sum(dataset)/float(len(dataset));

def variance(dataset):
	avg = average(dataset)
	var_sum = 0

	for d in dataset:
		var_sum += pow(d - avg, 2);
	
	return var_sum/float(len(dataset));

def print_stats(stats):	
	lengths = []
	times = []

	for s in stats:
		lengths.append(s[0])
		times.append(s[1])

	print("Average length : %s" % average(lengths))
	print("Variance length : %s" % variance(lengths))
	print("Average time : %s" % average(times))
	print("Variance time : %s" % variance(times))

start_time = time.time()
simulations = 1000
size = 4
nbcolors = 7

stats = []

for i in xrange(simulations):
	stats.append(single_game(size, nbcolors))

print("Board of size %s which %s different colors" % (size, nbcolors))
print_stats(stats)
print("--- %s seconds ---" % (time.time() - start_time))
