from util import load_example_data
from mlp import MLP

Xtrain, Ytrain, Xtest, Ytest = load_example_data(split=True)

model = MLP()
model.fit(Xtrain, Ytrain, [200, 200], adam=True)
model.plot_cost()

pred = model.predict(Xtest)
score = model.score(pred, Ytest)
print("Final classification rate with adam: {:.2f}".format(score))