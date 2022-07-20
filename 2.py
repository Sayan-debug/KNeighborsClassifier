from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
iris = datasets.load_iris()
features = iris.data
labels = iris.target
print(iris.DESCR)
print(features[0], labels[0])

# tranning the classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)
print(clf.predict([[1, 2, 3, 4]]))
