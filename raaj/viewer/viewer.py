import pyperception_lib as pylib
import numpy as np
import time
import threading

# #cloud = np.ones((5,9)).astype(np.float32)
#

#
#
# pylib.test(cloud)


class Visualizer(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        self.kill_received = False

        self.visualizer = pylib.Visualizer()
 
    def run(self):
        self.visualizer.start();

        while not self.kill_received:
            # your code
            #print self.name, "is active"
            self.visualizer.loop()
            time.sleep(0.1)

    def addCloud(self, cloud, size=1):
        self.visualizer.addCloud(cloud, size)

    def swapBuffer(self):
        self.visualizer.swapBuffer()

if __name__ == '__main__':

    visualizer = Visualizer("V")
    visualizer.start()

    cloud = np.load("../all_together.npy").T

    cloud[:,0:3] = 1

    print(cloud)



    # cloud = np.array([
    #     [0,0,0, 250,0,0, 0,0,0],
    #     [0,0,1, 1,250,0, 0,0,0],
    #     [0,0,2, 1,250,0, 0,0,0],
    # ]).astype(np.float32)

    while 1:
        time.sleep(0.001)

        cloud += 0.0001

        visualizer.addCloud(cloud, 5)
        visualizer.swapBuffer()

