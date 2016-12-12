from network import network
import sys


def main():
    if len(sys.argv) < 4:
        print("required arguments: path, no_of_epochs, batchsize")
    net = network()
    net.train(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

'''
def main():
    net = network()
    net.test(sys.argv[1])    
'''

if __name__ == "__main__":
    main()
