import numpy as np
import random
import math
from collections import namedtuple

Point = namedtuple("Point", ['x', 'y'])
random.seed(123)

def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    initial_solution = list(range(0, nodeCount))
    solution = initial_solution

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])
    new_obj = np.inf
    initial_obj = obj
    temperature = 10000
    alpha = 0.9995
    stopping_iter = 20000
    iter = 0
    while  iter < stopping_iter:
        t = random.randint(0, len(points) - 1)
        j = random.randint(0, len(points) - 1)
        new_solution = []
        if t > j:
            for count in range(0, j):
                new_solution.append(solution[count])
            for count in range(t, j - 1, -1):
                new_solution.append(solution[count])
            for count in range(t + 1, len(solution)):
                new_solution.append(solution[count])
        else:
            for count in range(0, t):
                new_solution.append(solution[count])
            for count in range(j, t - 1, -1):
                new_solution.append(solution[count])
            for count in range(j + 1, len(solution)):
                new_solution.append(solution[count])
        new_obj = length(points[new_solution[-1]], points[new_solution[0]])
        for index in range(0, nodeCount - 1):
            new_obj += length(points[new_solution[index]], points[new_solution[index + 1]])
        if new_obj < obj:
            obj = new_obj
            solution = new_solution
        else:
            a = np.random.choice([0, 1], p=[1 - math.exp(-(new_obj - obj) / temperature), math.exp(-(new_obj - obj) / temperature)])
            if a == 1:
                solution = new_solution
                obj = new_obj
        if initial_obj > obj:
            initial_obj = obj
            initial_solution = solution
        temperature = temperature * alpha
        if temperature < 1:
            temperature += 2
        iter += 1

    def reverse_segment_if_better(tour, i, j, k):
        """If reversing tour[i:j] would make the tour shorter, then do it."""
        # Given tour [...A-B...C-D...E-F...]
        A, B, C, D, E, F = tour[i - 1], tour[i], tour[j - 1], tour[j], tour[k - 1], tour[k % len(tour)]
        d0 = length(points[A], points[B]) + length(points[C], points[D]) + length(points[E], points[F])
        d1 = length(points[A], points[C]) + length(points[B], points[D]) + length(points[E], points[F])
        d2 = length(points[A], points[B]) + length(points[C], points[E]) + length(points[D], points[F])
        d3 = length(points[A], points[D]) + length(points[E], points[B]) + length(points[C], points[F])
        d4 = length(points[F], points[B]) + length(points[C], points[D]) + length(points[E], points[A])

        if d0 > d1:
            tour[i:j] = reversed(tour[i:j])
            return -d0 + d1
        elif d0 > d2:
            tour[j:k] = reversed(tour[j:k])
            return -d0 + d2
        elif d0 > d4:
            tour[i:k] = reversed(tour[i:k])
            return -d0 + d4
        elif d0 > d3:
            tmp = tour[j:k] + tour[i:j]
            tour[i:k] = tmp
            return -d0 + d3
        return 0

    def three_opt(tour):
        while True:
            delta = 0
            for (a, b, c) in all_segments(len(tour)):
                delta += reverse_segment_if_better(tour, a, b, c)
            if delta >= 0:
                break
        return tour

    def all_segments(n):
        return ((i, j, k)
                for i in range(n)
                for j in range(i + 1, n)
                for k in range(j + 1, n))

    # build a trivial solution
    # visit the nodes in the order they appear in the file

    # calculate the length of the tour
    for k in range(1):
        solution = three_opt(initial_solution)
        obj = length(points[solution[-1]], points[solution[0]])
        for index in range(0, nodeCount - 1):
            obj += length(points[solution[index]], points[solution[index + 1]])


    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


input_data_file = open('1.txt', 'r')
input_data = input_data_file.read()
print(solve_it(input_data))