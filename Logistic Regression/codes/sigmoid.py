import numpy as np
import matplotlib.pyplot as plt
import scipy

x = np.linspace(-5,5,30)
def sigmoid(x):
    return 1/(1+np.exp(-x))
vec_sigmoid = scipy.vectorize(sigmoid)


plt.plot(x,vec_sigmoid(x))
plt.xticks(np.arange(-5,6,1))
plt.yticks(np.arange(0,1.1,0.1))
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.grid()
plt.axhline(y=0.5,color = 'r',linestyle = 'dashed')
plt.text(2,0.45,"Decision Boundary",fontsize=11)
plt.savefig("sigmoid.png")
