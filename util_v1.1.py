import heapq
import numpy as np
from typing import Tuple

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
    vfunc = vectorize(mapper)
    return vfunc(mtx)


class mazeProblem:
    # initialise requires map. Can take start and end positions as tuplies
    def __init__(self, map, start: Tuple[int, int], end: Tuple[int, int]):
        self.map = np.array(map)
        self.start = start if start else (0, 0)
        self.end = end if end else map.shape
        self.loc = self.start

    def isStart(self):
        if self.loc == = self.start:
            return True
        return False

    def isEnd(self):
        if self.loc == = self.end:
            return True
        return False

    def moveTo(self, loc: tuple):
        self.loc = loc

    # gets valid moves based on current position, returns list of tuples
    # checks that moves are within bounds and are on a TRUE
    def getMoves(self):
        curr_i = self.loc[0]
        curr_j = self.loc[1]
        moves = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                to_i = curr_i + i
                to_j = curr_j + j
                if i == j == 0:
                    continue
                elif not (0<=to_i<len(self.map) and 0<=to_j<len(self.map[0])):
                    continue
                elif self.map[to_i][to_j]:
                    moves.append((to_i, to_j))
        return moves
    
def AStarMap(problem):
    map = problem.map
    fromMap = [[[] for j in range(len(map[0]))] for i in range(len(map))]
    





# for testing
def main():
    import random
    l1 = [random.randint(0, 20) for i in range(10)]
    l2 = minHeap(l1)
    print(l1)
    print(l2.heap)
    print(l2.heapSort())
    popped = l2.pop()
    print(popped)
    print(l2.heap)
    print(l2)
    print(len(l2))


if __name__ == "__main__":
    main()
