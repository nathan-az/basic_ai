import util
import heapq
import numpy as np
from typing import Tuple
import math
import random

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
    return np.abs(node1.x - node1.y) + np.abs(node2.x - node2.y)


def euclidDist(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)


def AStar(problem, h):
    # starting visualisation grid
    vis_grid = problem.obstacleMap.copy()
    vis_grid = np.vectorize(lambda x: "X" if x else "-")(vis_grid)


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
    heapq.heappush(toVisit, currNode)
    while (len(toVisit) > 0) and not problem.isEnd(currNode):
        # re-sort heap so first value is always smallest
        # may be able to avoid this step if we always update globalCost per node before pushing to heap
        # test after
        heapq.heapify(toVisit)

        # pop nodes that have been visited
        while (len(toVisit) > 0 and toVisit[0].visited):
            heapq.heappop(toVisit)
        if len(toVisit) < 1:
            break

        # we explore the first value in the toVisit array
        # this will be the value with the lowest globalCost as this as highest priority
        currNode = toVisit[0]
        # set it to visited so it is popped either next iteration or next time it is on the front of heap
        # probably could just pop above into currNode
        currNode.visited = True

        for nb in currNode.neighbours:
            # test after, checking localCost before heappush to keep order
            if not(nb.visited or nb.isObstacle):
                heapq.heappush(toVisit, nb)

            # evaluation cost Cost(nb) = pastCost(nb) + h(nb)
            costFromCurr = currNode.localCost + euclidDist(currNode, nb)

            if costFromCurr < nb.localCost:
                # set parent to the current node if it is nearest path to node
                nb.parent = currNode
                nb.localCost = costFromCurr
                nb.globalCost = nb.localCost + h(nb, endNode)

    # we now reverse the above search to print the path from start to end
    path = list()
    currNode = endNode
    path.append(currNode)
    if currNode.parent is None:
        print("No solution given obstacles")
        print(vis_grid)
        return
    while not problem.isStart(currNode):
        currNode = currNode.parent
        path.append(currNode)

    step = 0
    print("Step #:\t(x, y)")
    for i in range(len(path)-1, -1, -1):
        print("Step {}:\t({}, {})".format(step, path[i].x, path[i].y))
        step+=1
    
    print(vis_grid)
    return


obsMap = gen_2d_map(8,8,0.8)
problem = mazeSetup(obsMap, (1,0), (7,7))
AStar(problem, manhDist)