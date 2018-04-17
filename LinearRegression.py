import numpy
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

iris = pd.read_csv('93cars.csv')
X = iris['Weight'].values[:, numpy.newaxis]  #DependentVariable
Y = iris['EngineSize'].values
LinReg = LinearRegression()
# print "Actual", X[:10]   # For Printing first 10 Values
LinReg.fit(X, Y)             # Training
pre = LinReg.predict(X)      # Predicting
error=abs(X-pre)
errorsq=error.dot(error)
er = numpy.sum((pre - Y) ** 2)
SquaredError = numpy.sqrt(er/len(pre))
print "Pearson's Correlation Coefficient = ",LinReg.coef_
# print mse
print "Sum of squared errors = ",er
# print pre
print "R2 or Accuracy = ", LinReg.score(X, Y)   # Accuracy
print "y = ", LinReg.coef_, "x + ", LinReg.intercept_

# print "Predicted", pre[:10]
plt.scatter(X, Y)
plt.xlabel('Weight')
plt.ylabel('EngineSize')
plt.title('Prediction of Weight acc to Engine Size')
plt.plot(X, pre)
plt.show()