import heapq
import numpy as np
from typing import Tuple
import math
import random

# creating minHeap class for A*


class minHeap:
    # fixes minheap at location i
    def minHeapify(self, arr, i):
        l = 2*i+1
        r = l+1
        smallest = i
        n = len(arr)
        if l < n:
            if arr[l] < arr[smallest]:
                smallest = l
        if r < n:
            if arr[r] < arr[smallest]:
                smallest = r
        if smallest != i:
            arr[i], arr[smallest] = arr[smallest], arr[i]
            self.minHeapify(arr, smallest)

    # l and r for up to size//2-1 will have to maintain minheap property, iterate, minheapify
    def buildMinHeap(self, arr):
        for i in range(len(arr)//2-1, -1, -1):
            self.minHeapify(arr, i)

    # probably not going to need to sort for A*, still included sort for completion
    def heapSort(self):
        res = []
        copy = self.heap.copy()
        heap_size = len(copy)
        while heap_size > 0:
            copy[0], copy[heap_size-1] = copy[heap_size-1], copy[0]
            res.append(copy.pop())
            self.minHeapify(copy, 0)
            heap_size -= 1
        return res

    # takes an existing array, copies, builds a minheap
    def __init__(self, l1):
        self.heap = l1.copy()
        self.buildMinHeap(self.heap)

    # pops head, then fixes minheap property
    def pop(self):
        heap = self.heap
        heap_size = len(heap)
        if heap_size > 0:
            heap[0], heap[heap_size - 1] = heap[heap_size - 1], heap[0]
            res = heap.pop()
            self.minHeapify(heap, 0)
        return res

    # dunders below shouldn't be needed but are convenient for testing
    def __len__(self):
        return len(self.heap)

    def __repr__(self):
        print(self.heap)

    def __str__(self):
        return "Min heap of size {}: {}".format(len(self), self.heap)

# generate a 2d map of trues and false


def gen_2d_map(m, n, p):
    mtx = np.random.random((m, n))

    def mapper(x): return True if x > p else False
    vfunc = np.vectorize(mapper)
    return vfunc(mtx)


class mNode:
    def __init__(self, isObstacle=False, visited=False, globalDist=float('inf'),
                 locDist=float('inf'), x=None, y=None, neighbours=[], parent=None):
        self.isObstacle = isObstacle
        self.visited = visited
        self.globalDist = globalDist
        self.locDist = locDist
        self.x = x
        self.y = y
        self.neighbours = neighbours
        self.parent = parent


class mazeSetup:
    # initialise requires map. Can take start and end positions as tuples
    def createNodeMap(self, obstacleMap):
        height, width = obstacleMap.shape
        nodeMap = [[mNode(isObstacle=obstacleMap[i, j], x=j, y=i)
                    for j in range(width)] for i in range(height)]
        for i in range(height):
            for j in range(width):
                for ud in range(-1, 2):
                    for lr in range(-1, 2):
                        if 0 <= (i+ud) < height and 0 <= (j + lr) < width and not(lr == ud == 0):
                            nodeMap[i][j].neighbours.append(
                                nodeMap[i+ud][j+lr])
        return nodeMap

    # takes start and end as x,y coordinates
    def __init__(self, obstacleMap, start: Tuple[int, int], end: Tuple[int, int]):
        self.map = self.createNodeMap(obstacleMap)
        self.start = start
        self.end = end
        self.map[start[1]][start[0]].isObstacle = False
        self.map[end[1]][start[0]].isObstacle = False

    def isStart(self, node):
        if (node.x == self.start[0]) and (node.y == self.start[1]):
            return True
        return False

    def isEnd(self, node):
        if (node.x == self.end[0]) and (node.y == self.end[1]):
            return True
        return False

# calculates the manhattan distance between two nodes requiring x and y fields
def nodeManhattanDistance(node1, node2):
    return np.abs(node1.x - node1.y) + np.abs(node2.x - node2.y)



# for testing
def main():
    pass


if __name__ == "__main__":
    map = gen_2d_map(10, 15, 0.3)
    problem = mazeSetup(map, (3, 3), (5, 3))
    print(problem.map)