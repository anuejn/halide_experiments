from time import time

def benchmark(callback):
    start = time()
    end = 0

    count = 0
    while True:
        callback()
        count += 1
        end = time()
        if end - start > 1:
            break

    print(f"{count} iterations in {(end - start)}s = {count / (end - start)}fps")