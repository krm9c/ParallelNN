import multiprocessing
from multiprocessing import Process
import os

def info(title):
    print title
    print 'module name:', __name__
    if hasattr(os, 'getppid'):
        print 'parent process:', os.getppid()
    print 'process id:', os.getpid()
    print("\n")

def f(name):
    print 'hello', name
    info('function f')
    

if __name__ == '__main__':
    info('main line')
    name = ['bob', 'alice', 'john']
    for i in name:
        print i
        p = Process(target=f, args=(i,))
        p.start()
        p.join()