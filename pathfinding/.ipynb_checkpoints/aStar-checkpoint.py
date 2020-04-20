import heapq
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from typing import Tuple
from random import random

# generate a 2d map of trues and false


def generate_2d_map(m, n, p):
    matrix = np.random.random((m, n))

    def mapper(x): return True if x > p else False
    vfunc = np.vectorize(mapper)
    return vfunc(matrix)

# MapNode class (map node) stores required information for the search


class MapNode:
    def __init__(self, is_obstacle=False, visited=False, global_cost=float('inf'),
                 local_cost=float('inf'), x=None, y=None, neighbours=list(), parent=None):
        self.is_obstacle = is_obstacle
        self.visited = visited
        self.global_cost = global_cost
        self.local_cost = local_cost
        self.x = x
        self.y = y
        self.neighbours = neighbours
        self.parent = parent

    def __lt__(self, other):
        return self.global_cost < other.global_cost

    def __eq__(self, other):
        return self.global_cost == other.global_cost


class ProblemSetup:
    # initialise requires map. Can take start and end positions as tuples
    def create_node_map(self, obstacle_map):
        height, width = obstacle_map.shape
        # set is_obstacle attribute based on index in obstacle_map
        node_map = [[MapNode(is_obstacle=obstacle_map[i, j], x=j, y=i, neighbours=list())
                    for j in range(width)] for i in range(height)]
        # allows diagonal movement, sets array of neighbouring nodes
        for i in range(height):
            for j in range(width):
                current_node = node_map[i][j]
                if self.allow_diagnoal:
                    # up down, left right, for finding the potential 8 neighbouring nodes
                    for ud in range(-1, 2):
                        for lr in range(-1, 2):
                            if 0 <= (i+ud) < height and 0 <= (j+lr) < width and (not(lr == ud == 0)):
                                current_node.neighbours.append(
                                    node_map[i+ud][j+lr])
                else:
                    if 0 <= (i + -1):
                        current_node.neighbours.append(node_map[i + -1][j])
                    if 0 <= (j + -1):
                        current_node.neighbours.append(node_map[i][j + -1])
                    if (i + 1) < height:
                        current_node.neighbours.append(node_map[i + 1][j])
                    if (j + 1) < width:
                        current_node.neighbours.append(node_map[i][j + 1])

        return node_map

    # takes start and end as x,y coordinates
    def __init__(self, obstacle_map, start: Tuple[int, int], end: Tuple[int, int], allow_diagnoal=False):
        self.obstacle_map = obstacle_map
        self.allow_diagnoal = allow_diagnoal
        self.obstacle_map[start[1]][start[0]] = False
        self.obstacle_map[end[1]][end[0]] = False
        self.nMap = self.create_node_map(obstacle_map)
        self.start = start
        self.end = end

    def is_start(self, node):
        if (node.x == self.start[0]) and (node.y == self.start[1]):
            return True
        return False

    def is_end(self, node):
        if (node.x == self.end[0]) and (node.y == self.end[1]):
            return True
        return False

# calculates the manhattan distance between two nodes requiring x and y fields


def calculate_manhattan_distance(node1, node2):
    return np.abs(node1.x - node2.x) + np.abs(node1.y - node2.y)


def calculate_euclidean_distance(node1, node2):
    return sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)


def AStar(problem, h):
    # set starting and ending nodes
    start_node = problem.nMap[problem.start[1]][problem.start[0]]
    end_node = problem.nMap[problem.end[1]][problem.end[0]]

    # distance from start to starting node is zero
    start_node.local_cost = 0

    # best guess of cost from starting node to end is heuristic
    start_node.global_cost = h(start_node, end_node)

    # pointer for current node to starting node for iteration condition
    current_node = start_node

    to_visit = list()
    visited = list()
    heapq.heappush(to_visit, current_node)
    while (len(to_visit) > 0) and not problem.is_end(current_node):
        # re-sort heap so first value is always smallest
        # may be able to avoid this step if we always update global_cost per node before pushing to heap
        # test after
        heapq.heapify(to_visit)

        # we explore the first value in the to_visit array
        # this will be the value with the lowest global_cost as this as highest priority
        current_node = heapq.heappop(to_visit)
        visited.append(current_node)
        # set it to visited so it is popped either next iteration or next time it is on the front of heap
        current_node.visited = True

        for neighbour in current_node.neighbours:
            if not(neighbour.visited or neighbour.is_obstacle):
                heapq.heappush(to_visit, neighbour)

            # evaluation cost Cost(s) = pastCost(s) + h(s)
            cost_from_current = current_node.local_cost + calculate_euclidean_distance(current_node, neighbour)

            if cost_from_current < neighbour.local_cost:
                # set parent to the current node if it is nearest path to node
                neighbour.parent = current_node
                neighbour.local_cost = cost_from_current
                neighbour.global_cost = neighbour.local_cost + h(neighbour, end_node)

    return start_node, end_node, visited, to_visit


def optimal_path(start_node, end_node):
    path = list()
    current_node = end_node
    path.append(current_node)
    if current_node.parent is None:
        return path
    while current_node.parent is not None:
        current_node = current_node.parent
        path.append(current_node)
    return path


def print_path_list(path):
    if len(path) < 1:
        print("There is no path from start to end")
        return
    print("#:\t(x, y)")
    step = 0
    for i in range(len(path)-1, -1, -1):
        print("{}:\t({}, {})".format(step, path[i].x, path[i].y))
        step += 1



# below function prints the results to two matplotlib graphics
def display_problem(path, start_node, end_node, visited, to_visit, obstacle_map):
    # sets colours for start and end
    target_vis = obstacle_map.copy()
    target_vis = np.vectorize(lambda x: 100 if x else 0)(target_vis)
    target_vis[start_node.y][start_node.x] = 10
    target_vis[end_node.y][end_node.x] = 20

    # colour map and assignment
    cmap = colors.ListedColormap(
        ['white', "xkcd:robin's egg blue", 'xkcd:very light green', 'black', 'xkcd:crimson', 'grey'])
    bounds = [-1, 1, 3, 5, 15, 50, 150]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(figsize=(10, 10*1.5*len(target_vis)/len(target_vis[0])))
    ax0 = fig.add_subplot(211)
    ax = fig.add_subplot(212)

    ax0.imshow(target_vis, cmap=cmap, norm=norm)
    ax0.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2)
    ax0.set_xticks(np.arange(-0.5, len(target_vis[0]), 1))
    ax0.set_yticks(np.arange(-0.5, len(target_vis), 1))
    ax0.title.set_text("Search problem")

    for node in to_visit:
        target_vis[node.y][node.x] = 4
    for node in visited:
        target_vis[node.y][node.x] = 2

    target_vis[start_node.y][start_node.x] = 10
    target_vis[end_node.y][end_node.x] = 20

    if len(path) < 2:
        ax.imshow(target_vis, cmap=cmap, norm=norm)
        ax.grid(which="major", axis="both", linestyle="-",
                marker="bo", color="k", linewidth=2)
        ax.set_xticks(np.arange(-0.5, len(target_vis[0]), 1))
        ax.set_yticks(np.arange(-0.5, len(target_vis), 1))
        ax.title.set_text("A* found no solution")
        plt.show()
        return

    xs = [node.x for node in path]
    ys = [node.y for node in path]
    print(xs, ys, sep="\n")

    ax.plot(xs, ys, color="yellow", marker="o", linestyle="-")

    ax.title.set_text("A* solution")
    ax.imshow(target_vis, cmap=cmap, norm=norm)
    ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2)
    ax.set_xticks(np.arange(-0.5, len(target_vis[0]), 1))
    ax.set_yticks(np.arange(-0.5, len(target_vis), 1))
    plt.show()
    return

# below function has been superseded. It was the first attempt at visualising the output
def print_char_display(path, start_node, end_node, visited, to_visit, obstacle_map):
    print("Pathfinding problem:")
    target_vis = obstacle_map.copy()
    target_vis = np.vectorize(lambda x: 'x' if x else ' ')(target_vis)
    target_vis[start_node.y][start_node.x] = "S"
    target_vis[end_node.y][end_node.x] = "E"
    for row in target_vis:
        print("|{}|".format("  ".join(row)))

    for node in visited:
        target_vis[node.y][node.x] = "*"
    for node in to_visit:
        target_vis[node.y][node.x] = '-'

    target_vis[start_node.y][start_node.x] = "S"
    target_vis[end_node.y][end_node.x] = "E"

    if len(path) < 2:
        print("\nThere is no path from start to end.\nNode exploration:")
        for row in target_vis:
            print("|{}|".format("  ".join(row)))
        return

    for i in range(1, len(path)-1):
        target_vis[path[i].y][path[i].x] = "o"
    print("\nA* optimal path solution:")
    for row in target_vis:
        print("|{}|".format("  ".join(row)))
    return
