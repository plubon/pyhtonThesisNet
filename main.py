from network import network
import sys


def main(_):
    if len(sys.argv) < 4:
        print("required arguments: path, no_of_epochs, batchsize")
    net = network()
    net.train(sys.argv[1], sys.argv[2], sys.argv[3])


