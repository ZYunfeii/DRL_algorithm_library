from multiprocessing import Process, Manager, Pipe
from threading import Thread
import time

# def main():
#     data = []
#     t1 = Thread(target=func,args=(data,))
#     t1.start()
#     print(data)
#     t1 = Thread(target=func, args=(data,))
#     t1.start()
#     print(data)
#
# def func(data):
#     data.append(1)
def main():
    data = []
    pipe1,pipe2 = Pipe()
    p1 = Process(target=func,args=(pipe2,))
    pipe1.send((1,))
    p1.start()
    time.sleep(1)

def func(pipe2):
    print(pipe2.recv()[0])


if __name__ == "__main__":
    main()