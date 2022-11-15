# %%
from MLLib import *
import random

testFL = FL(sigmoid, sigmoid_prime)
X1 = np.random.rand(5)
Y1 = sigmoid(X1)
Y_ideal1 = np.random.rand(5)
ΔX1 = testFL.cost_derivative(Y1, Y_ideal1)


Y2 = testFL.push(X1)
assert Y2.all() == Y1.all()

ΔX2 = testFL.acceptFeedback(Y_ideal1)
assert ΔX2.all() == ΔX1.all()