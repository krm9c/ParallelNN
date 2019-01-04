from multiprocessing import Process, Pipe
from multiprocessing import Pool

import os

def info(title):
    print title
    print 'module name:', __name__
    if hasattr(os, 'getppid'):
        print 'parent process:', os.getppid()
    print 'process id:', os.getpid()
 

def f(conn,num):
    message = ["Hello", "Tata", "Alvida"]
    info(conn)
    conn.send([42, None, message[num]])
    conn.close()    


if __name__ == '__main__':
    for i in range(3):
        parent_conn, child_conn = Pipe()
        p = Process(target=f, args=(child_conn,i,))
        p.start()
        print parent_conn.recv()
        p.join()
        print("\n")
    
'''  message = ["Hello", "Tata", "Alvida"]

    for i in message: 
        parent_conn.send([i])
   
    for i in range(1,4):
        print child_conn.recv()
        info("Child")

 '''