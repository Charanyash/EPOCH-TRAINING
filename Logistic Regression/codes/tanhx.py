import numpy as np
import matplotlib.pyplot as plt
import scipy

def Tanhx(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))

X = np.linspace(-10,10,50)
vec_Tanhx = scipy.vectorize(Tanhx)

plt.plot(X,vec_Tanhx(X))
plt.xlabel("$x$")
plt.ylabel("$Tanhx$")
plt.grid()
plt.axhline(y=0,color = 'r',linestyle="dotted")
plt.axhline(y=1,linestyle = 'dashed')
plt.axhline(y=-1,linestyle = "dashed")
plt.text(0.5,0.05,"DecisionBoundary")
plt.text(8,0.9,"$y = 1$")
plt.text(-8,-0.9,"$y=-1$")
plt.savefig("altsigmoid.png")