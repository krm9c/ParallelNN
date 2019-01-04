from multiprocessing import Process, Queue

import os

def info(title):
    print title
    print 'module name:', __name__
    if hasattr(os, 'getppid'):
        print 'parent process:', os.getppid()
    print 'process id:', os.getpid()

def f(q):
    info(q)
    q.put([42, None, 'hello'])

if __name__ == '__main__':
    for i in range(3):
        q = Queue()
        p = Process(target=f, args=(q,))
        p.start()
        print q.get()
        p.join()
        print("\n")

        