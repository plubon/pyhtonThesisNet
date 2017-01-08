import tensorflow as tf
import numpy as np
import sys
from datahelper import datahelper


def main():
    if len(sys.argv) < 3:
        print("required arguments: path to images, path to model")
        exit()
    print(sys.argv[2])
    sess = tf.InteractiveSession()
    new_saver = tf.train.import_meta_graph(sys.argv[2] + '.meta')
    new_saver.restore(sess, sys.argv[2])
    dh = datahelper(sys.argv[1])
    testdata = dh.gettestdata()
    finAcc = 0
    test_len = 0
    for batch in testdata:
        acc = sess.run(tf.get_default_graph().get_tensor_by_name("Accuracy:0"),
                      feed_dict={'DefaultInput:0': batch.data, 'DropoutRate:0': 1, 'Labels:0': batch.labels})
        finAcc += acc * len(batch.data)
        test_len += len(batch.data)
    finAcc = finAcc / test_len
    print(finAcc)

if __name__ == "__main__":
    main()
