import multiprocessing
import subprocess
import os
from multiprocessing import Process, Value, Array, Lock

arr = Array('i', range(10) )
arr[0] = 2

def info(title):
    print title
    print 'module name:', __name__
    if hasattr(os, 'getppid'):
        print 'parent process:', os.getppid()
    print 'process id:', os.getpid()
    print("\n")

import multiprocessing

def worker(d, key,l):
    l.acquire()
    value = key*3
    d[key] = value
    arr[key] = value*2
    l.release()

if __name__ == '__main__':
    mgr = multiprocessing.Manager()
    d = mgr.dict()
    lock = Lock()
    jobs = [ multiprocessing.Process(target=worker, args=(d, i, lock))
             for i in range(1,10) 
             ]
    for j in jobs:
        j.start()
        print info(j)
    for j in jobs:
        j.join()
    print 'Results:', d
    print arr[:]