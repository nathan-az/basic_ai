import util
import heapq
import numpy as np
from typing import Tuple
import math
import random
import matplotlib.pyplot as plt
from matplotlib import colors

# generate a 2d map of trues and false


def gen_2d_map(m, n, p):
    mtx = np.random.random((m, n))

    def mapper(x): return True if x > p else False
    vfunc = np.vectorize(mapper)
    return vfunc(mtx)

# mNode class (map node) stores required information for the search
class mNode:
    def __init__(self, isObstacle=False, visited=False, globalCost=float('inf'),
                 localCost=float('inf'), x=None, y=None, neighbours=list(), parent=None):
        self.isObstacle = isObstacle
        self.visited = visited
        self.globalCost = globalCost
        self.localCost = localCost
        self.x = x
        self.y = y
        self.neighbours = neighbours
        self.parent = parent

    def __lt__(self, other):
        return self.globalCost < other.globalCost

    def __eq__(self, other):
        return self.globalCost == other.globalCost


class mazeSetup:
    # initialise requires map. Can take start and end positions as tuples
    def createNodeMap(self, obstacleMap):
        height, width = obstacleMap.shape
        # set isObstacle attribute based on index in obstacleMap
        nodeMap = [[mNode(isObstacle=obstacleMap[i, j], x=j, y=i, neighbours=list())
                    for j in range(width)] for i in range(height)]

        # allows diagonal movement, sets array of neighbouring nodes
        for i in range(height):
            for j in range(width):
                currNode = nodeMap[i][j]
                for ud in range(-1, 2):
                    for lr in range(-1, 2):
                        if 0 <= (i+ud) < height and 0 <= (j+lr) < width and (not(lr == ud == 0)):
                            currNode.neighbours.append(
                                nodeMap[i+ud][j+lr])
        return nodeMap

    # takes start and end as x,y coordinates
    def __init__(self, obstacleMap, start: Tuple[int, int], end: Tuple[int, int]):
        self.obstacleMap = obstacleMap
        self.obstacleMap[start[1]][start[0]] = False
        self.obstacleMap[end[1]][end[0]] = False
        self.nMap = self.createNodeMap(obstacleMap)
        self.start = start
        self.end = end

    def isStart(self, node):
        if (node.x == self.start[0]) and (node.y == self.start[1]):
            return True
        return False

    def isEnd(self, node):
        if (node.x == self.end[0]) and (node.y == self.end[1]):
            return True
        return False

# calculates the manhattan distance between two nodes requiring x and y fields


def manhDist(node1, node2):
    return np.abs(node1.x - node2.x) + np.abs(node1.y - node2.y)


def euclidDist(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def AStar(problem, h):
    # set starting and ending nodes
    startNode = problem.nMap[problem.start[1]][problem.start[0]]
    endNode = problem.nMap[problem.end[1]][problem.end[0]]

    # distance from start to starting node is zero
    startNode.localCost = 0

    # best guess of cost from starting node to end is heuristic
    startNode.globalCost = h(startNode, endNode)

    # pointer for current node to starting node for iteration condition
    currNode = startNode

    toVisit = list()
    visited = list()
    heapq.heappush(toVisit, currNode)
    while (len(toVisit) > 0) and not problem.isEnd(currNode):
        # re-sort heap so first value is always smallest
        # may be able to avoid this step if we always update globalCost per node before pushing to heap
        # test after
        heapq.heapify(toVisit)

        if len(toVisit) < 1:
            break

        # we explore the first value in the toVisit array
        # this will be the value with the lowest globalCost as this as highest priority
        currNode = heapq.heappop(toVisit)
        visited.append(currNode)
        # set it to visited so it is popped either next iteration or next time it is on the front of heap
        currNode.visited = True

        for nb in currNode.neighbours:
            if not(nb.visited or nb.isObstacle):
                heapq.heappush(toVisit, nb)

            # evaluation cost Cost(nb) = pastCost(nb) + h(nb)
            costFromCurr = currNode.localCost + euclidDist(currNode, nb)

            if costFromCurr < nb.localCost:
                # set parent to the current node if it is nearest path to node
                nb.parent = currNode
                nb.localCost = costFromCurr
                nb.globalCost = nb.localCost + h(nb, endNode)

    return startNode, endNode, visited, toVisit

def optimal_path(startNode, endNode):
    path = list()
    currNode = endNode
    path.append(currNode)
    if currNode.parent is None:
        return path
    while currNode != startNode:
        currNode = currNode.parent
        path.append(currNode)
    return path

def print_path_list(path):
    if len(path)<1:
        print("There is no path from start to end")
        return
    print("#:\t(x, y)")
    step = 0
    for i in range(len(path)-1, -1, -1):
        print("{}:\t({}, {})".format(step, path[i].x, path[i].y))
        step+=1

def print_visual(path, startNode, endNode, visited, toVisit, obsMap):
    print("Pathfinding problem:")
    target_vis = obsMap.copy()
    target_vis = np.vectorize(lambda x: 'x' if x else ' ')(target_vis)
    target_vis[startNode.y][startNode.x] = "S"
    target_vis[endNode.y][endNode.x] = "E"
    for row in target_vis:
        print("|{}|".format("  ".join(row)))

    for node in visited:
        target_vis[node.y][node.x] = "*"
    for node in toVisit:
        target_vis[node.y][node.x] = '-'

    target_vis[startNode.y][startNode.x] = "S"
    target_vis[endNode.y][endNode.x] = "E"

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


def print_grid(path, startNode, endNode, visited, toVisit, obsMap):
    target_vis = obsMap.copy()
    target_vis = np.vectorize(lambda x: 100 if x else 0)(target_vis)
    target_vis[startNode.y][startNode.x] = 10
    target_vis[endNode.y][endNode.x] = 20

    cmap = colors.ListedColormap(['white', 'xkcd:light blue', 'xkcd:very pale green', 'xkcd:royal blue', 'black', 'xkcd:crimson', 'grey'])
    bounds = [-1, 1, 3, 5, 8, 15, 50, 150]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(figsize=(10, 10*1.5*len(target_vis)/len(target_vis[0])))
    ax0 = fig.add_subplot(211)
    ax = fig.add_subplot(212)

    ax0.imshow(target_vis, cmap=cmap, norm=norm)
    ax0.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2)
    ax0.set_xticks(np.arange(-0.5, len(target_vis[0]), 1))
    ax0.set_yticks(np.arange(-0.5, len(target_vis), 1))
    ax0.title.set_text("Search problem")

    for node in visited:
        target_vis[node.y][node.x] = 2
    for node in toVisit:
        target_vis[node.y][node.x] = 4

    target_vis[startNode.y][startNode.x] = 10
    target_vis[endNode.y][endNode.x] = 20

    if len(path) < 2:
        ax.imshow(target_vis, cmap=cmap, norm=norm)
        ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2)
        ax.set_xticks(np.arange(-0.5, len(target_vis[0]), 1))
        ax.set_yticks(np.arange(-0.5, len(target_vis), 1))
        ax.title.set_text("A* found no solution")
        plt.show()
        

    for i in range(len(path)):
        target_vis[path[i].y][path[i].x] = 6
    target_vis[startNode.y][startNode.x] = 10
    target_vis[endNode.y][endNode.x] = 20


    ax.title.set_text("A* solution")
    ax.imshow(target_vis, cmap=cmap, norm=norm)
    ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2)
    ax.set_xticks(np.arange(-0.5, len(target_vis[0]), 1))
    ax.set_yticks(np.arange(-0.5, len(target_vis), 1))
    plt.show()
    return
