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


# for testing
def main():
    pass


if __name__ == "__main__":
    map = gen_2d_map(10, 15, 0.3)
    problem = mazeSetup(map, (3, 3), (5, 3))
    print(problem.map)