#testing file

#heapify does not take key for sorting classes, testing use with __lt__
def test_objHeap():
    import heapq
    import random

    class holder:
        def __init__(self, val):
            self.val = val
        def __lt__(self, other):
            return self.val < other.val
        def __str__(self):
            return self.val
        def __repr__(self):
            return str(self.val)

    arr = []
    heapq.heappush(arr, holder(3))
    heapq.heappush(arr, holder(5))
    heapq.heappush(arr, holder(1))
    print(arr)

if __name__ == "__main__":
    test_objHeap()