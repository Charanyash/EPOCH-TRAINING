import numpy as np
import matplotlib.pyplot as plt
import scipy

def sigmoid(x):
    return 1/(1+np.exp(-x))

def cost(h,y):
    prob = sigmoid(h)
    if y ==1 :
        return -np.log(prob)
    else:
        return -np.log(1-prob) 
vec_cost = scipy.vectorize(cost)
vec_sigmoid = scipy.vectorize(sigmoid)
h = np.linspace(-10,10,50)

plt.plot(sigmoid(h),cost(h,0))
plt.plot(sigmoid(h),cost(h,1))
plt.legend(["$Y=0$","$Y=1$"])
plt.grid()
plt.ylabel("Cost")
plt.xlabel("$prob(h(x))$")
plt.text(0.05,9,"$log(prob(h(x)))$")
plt.text(0.7,9,"$log(1-prob(h(x)))$")
plt.savefig("logi_cost.png")
plt.show()