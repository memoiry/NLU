import fasttext
classifier = fasttext.supervised('train.txt', 'model', epoch=25, lr = 1)
res = classifier.test('train.txt')
print(res.precision)
print(res.recall)