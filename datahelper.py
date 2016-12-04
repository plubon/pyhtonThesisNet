import os
import math
import numpy as np
from scipy import misc
import random

class datahelper:

    testProportion = 0.8

    def __init__(self, path):
        self.women = []
        self.men = []
        self.path = path
        labelsfile = open(path+os.sep+"labels", 'r')
        for line in labelsfile:
            tokens = line.split(";")
            if int(tokens[1]) == 0:
                self.men.append(tokens[0])
            else:
                self.women.append(tokens[0])
        self.womenTestIndex = math.ceil(len(self.women)*datahelper.testProportion)
        self.menTestIndex = math.ceil(len(self.men) * datahelper.testProportion)

    def getnextbatch(self, size):
        proportion = len(self.men) / (len(self.women) + len(self.men))
        menIndices = random.sample(range(0,  self.menTestIndex), math.ceil(proportion * size))
        womenIndices = random.sample(range(0, self.womenTestIndex), math.floor((1 - proportion) * size))
        arrays = []
        labels = []
        for idx in menIndices:
            frames = []
            labels.append([1, 0])
            for i in range(0, 25):
                image = misc.imread(self.path+os.sep+self.men[idx]+os.sep+str(i).zfill(3)+".jpg")
                frames.append(np.delete(image, 2, axis=2))
            arrays.append(np.concatenate(frames, 2))
        for idx in womenIndices:
            frames = []
            labels.append([0, 1])
            for i in range(0, 25):
                image = misc.imread(self.path + os.sep + self.men[idx] + os.sep + str(i).zfill(3) + ".jpg")
                frames.append(np.delete(image, 2, axis=2))
            arrays.append(np.concatenate(frames, 2))
        return batch(np.stack(arrays, 0), labels)


    def getsingledata(self):
        arrays=[]
        labels=[]
        frames = []
        labels.append([1, 0])
        for idx in range(1,26):
            image = misc.imread(self.path+os.sep+str(idx).zfill(3)+".jpg")
            frames.append(np.delete(image, 2, axis=2))
        arrays.append(np.concatenate(frames, 2))
        return batch(np.stack(arrays, 0), labels)
            

    def gettestdata(self):
        arrays = []
        labels = []
        for idx in range(self.menTestIndex, len(self.men)):
            frames = []
            labels.append([1, 0])
            for i in range(0, 25):
                image = misc.imread(self.path + os.sep + self.men[idx] + os.sep + str(i).zfill(3) + ".jpg")
                frames.append(np.delete(image, 2, axis=2))
            arrays.append(np.concatenate(frames, 2))
        for idx in range(self.womenTestIndex, len(self.women)):
            frames = []
            labels.append([0, 1])
            for i in range(0, 25):
                image = misc.imread(self.path + os.sep + self.men[idx] + os.sep + str(i).zfill(3) + ".jpg")
                frames.append(np.delete(image, 2, axis=2))
            arrays.append(np.concatenate(frames, 2))
        return batch(np.stack(arrays, 0), labels)


class batch:

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

