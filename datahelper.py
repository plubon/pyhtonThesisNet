import os
import math
import numpy as np
from scipy import misc
from scipy import ndimage
import random

class datahelper:

    testProportion = 0.9

    def __init__(self, path):
        self.women = []
        self.men = []
        self.womenTest = []
        self.menTest = []
        labelsfile = open(path + os.sep + "labels1", 'r')
        for line in labelsfile:
            tokens = line.split(";")
            dirc = path+os.sep+tokens[0]
            frame_count = len([name for name in os.listdir(dirc) if os.path.isfile(os.path.join(dirc, name))])
            print(frame_count)
            i = 0
            while (i + 25) < frame_count:
                frames = []
                for j in range(i, i+25):
                    image = misc.imread(path + os.sep + tokens[0] + os.sep + str(j).zfill(3) + ".png")
                    sums = image.sum(axis=0).sum(axis=0)
                    if sums[2] > sums[0] or sums[2] > sums[1]:
                        raise Exception("Something wrong with channels")
                    image = np.delete(image, 2, axis=2)
                    frames.append(image)
                if int(tokens[1]) == 0:
                    self.men.append(sample(frames, [1, 0]))
                elif int(tokens[1]) == 1:
                    self.women.append(sample(frames, [0, 1]))
                else:
                    raise Exception("Wrong label")
                i += 5
            dirc = path + os.sep + tokens[0]+"_1"
            frame_count = len([name for name in os.listdir(dirc) if os.path.isfile(os.path.join(dirc, name))])
            i = 0
            while (i + 25) < frame_count:
                frames = []
                for j in range(i, i + 25):
                    image = misc.imread(path + os.sep + tokens[0] + '_1' + os.sep + str(j).zfill(3) + ".png")
                    sums = image.sum(axis=0).sum(axis=0)
                    if sums[2] > sums[0] or sums[2] > sums[1]:
                        raise Exception("Something wrong with channels")
                    image = np.delete(image, 2, axis=2)
                    frames.append(image)
                if int(tokens[1]) == 0:
                    self.men.append(sample(frames, [1, 0]))
                elif int(tokens[1]) == 1:
                    self.women.append(sample(frames, [0, 1]))
                else:
                    raise Exception("Wrong label")
                i += 5
        self.womenTest = self.women[int(math.ceil(len(self.women)*self.testProportion)):]
        self.women = self.women[:int(math.floor(len(self.women)*self.testProportion))]
        self.menTest = self.men[int(math.ceil(len(self.men) * self.testProportion)):]
        self.men = self.men[:int(math.floor(len(self.men) * self.testProportion))]
        self.proportion = len(self.men) / float(len(self.women) + len(self.men))
        print("Proportion: " + str(self.proportion))
        print("No of men samples: " + str(len(self.men)))
        print("No of men test samples: " + str(len(self.menTest)))
        print("No of women samples: " + str(len(self.women)))
        print("No of women test samples: " + str(len(self.womenTest)))
        print(self.women[0].label)
        print(self.womenTest[0].label)
        print(self.men[0].label)
        print(self.menTest[0].label)

    def get_label(self, val):
        if int(val) == 1:
            return [0, 1]
        elif int(val) == 0:
            return [1, 0]
        else:
            raise Exception("Wrong label")


    def getnextbatch(self, size):
        men_indices = random.sample(range(len(self.men)), int(math.ceil(self.proportion * size)))
        women_indices = random.sample(range(len(self.women)), int(math.floor((1 - self.proportion) * size)))
        listsum = []
        arrays = []
        labels = []
        for idx in men_indices:
            listsum.append(self.men[idx])
        for idx in women_indices:
            listsum.append(self.women[idx])
        random.shuffle(listsum)
        for item in listsum:
            arrays.append(item.img)
            labels.append(item.label)
        return batch(np.stack(arrays, 0), labels)

    def getsingledata(self):
        arrays = [self.men[0].img]
        labels = [self.men[0].label]
        return batch(np.stack(arrays, 0), labels)
            

    def gettestdata(self):
        arrays = []
        labels = []
        listsum = self.menTest + self.womenTest
        random.shuffle(listsum)
        for item in listsum:
            arrays.append(item.img)
            labels.append(item.label)
        subarrays = [arrays[x:x + 100] for x in xrange(0, len(arrays), 100)]
        sublabels = [labels[x:x+100] for x in xrange(0, len(labels), 100)]
        ret = []
        for i in range(len(subarrays)):
            ret.append(batch(np.stack(subarrays[i], 0), sublabels[i]))
        return ret


class sample:
    def __init__(self, img, label):
        self.img = np.concatenate(img, 2)
        self.label = label


class batch:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

