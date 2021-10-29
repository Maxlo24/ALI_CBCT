from tqdm.std import tqdm

import time

size = 100_000_000

def sum(x):return x+x

# sum = lambda x: x+x
lst = []
val = range(size)
print("start")
start_time = time.time()
for i in val:
    lst.append(sum(i))
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
lst = map(sum,val)
print("--- %s seconds ---" % (time.time() - start_time))
