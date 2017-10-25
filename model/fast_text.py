import fasttext
import sys
import os

def fasttext_test(train, test, epoch=25):
    classifier = fasttext.supervised(train, 'model', epoch=epoch)
    res = classifier.test(test)
    print(res.precision)
    print(res.recall)
    return res.precision

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("pls use: python fast_text.py train test")
        sys.exit()
    train = sys.argv[1]
    test = sys.argv[2]
    fasttext_test(train, test)