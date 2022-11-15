# %%
from MLLib import *
import random
steps       = 5
XupperBound = 5
XlowerBound = -5
YupperBound = 30
YlowerBound = -1

testModel = EmpiricalFunction(steps, XupperBound, XlowerBound, YupperBound, YlowerBound, ['c1', 'c2'])
print(testModel)
# %%
ideal_params = np.array([(i-2)**2 for i in range(steps)])
idealModel = EmpiricalFunction(steps, XupperBound, XlowerBound, YupperBound, YlowerBound, ['c1', 'c2'])
idealModel.params = ideal_params


for i, m in enumerate(testModel):
    score = 0
    for i in range(10):
        x = random.random()*(XupperBound - XlowerBound) + XlowerBound
        y_ideal = idealModel(x)
        y = testModel(x)
        score = score + (y-y_ideal)

    m.pushScore({'c1':np.sqrt(i),'c2':-(m['v2']**2)+1})

print(testModel.updateToBest())
print()
