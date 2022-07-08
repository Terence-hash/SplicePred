import threading
import time
import os


count = 2


def sum1(x, y):
    t = threading.current_thread()
    for i in range(count):
        x[0] = y[0] + 1
        print(f"线程一({t.name})：{y[0]}")
        time.sleep(0.2)


def sum2(x, y):
    t = threading.current_thread()
    for i in range(count):
        y[0] = x[0] + 1
        print(f"线程二({t.name})：{x[0]}")
        time.sleep(0.2)


if __name__ == "__main__":
    print(f"pid={os.getpid()}")
    x = [0]
    y = [0]
    t1 = threading.Thread(target=sum1, name="thread-1", args=(x, y))
    t2 = threading.Thread(target=sum2, name="thread-2", args=(x, y))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
