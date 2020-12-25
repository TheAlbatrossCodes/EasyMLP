from util import load_example_data
from ann import ANN

Xtrain, Ytrain, Xtest, Ytest = load_example_data(split=True)

model = ANN()
model.fit(Xtrain, Ytrain, [200], adam=True)
model.plot_cost()

pred = model.predict(Xtest)
score = model.score(pred, Ytest)
print("Final classification rate with adam: {:.2f}".format(score))