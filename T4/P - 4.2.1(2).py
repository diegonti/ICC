### Exercise 4.2.1
import numpy as np
import matplotlib.pyplot as plt

#Definition of the integration functions (array-like operations)
def rectangularL(x,f): return h*sum(f(x))

def rectangularR(x,f): return h*sum(f(x+h))

def rectangularM(x,f): return h*sum(f((2*x+h)/2))

def trapezoid(x,f):
    points = f(x)
    return h*(sum((points[1:-1]))+(points[0]+points[-1])/2)

def simpson13(x,f):
    points = f(x)
    return (h/3)*(points[0] + 2*sum(points[2:-2:2]) + 4*sum(points[1:-1:2]) + points[-1])

def simpson23(x,f):
    points = f(x)
    return (3*h/8)*(sum(points[0:-3:3]) + 3*sum(points[1:-2:3]) + 3*sum(points[2:-1:3]) + sum(points[3::3]))

def Eabs(I,Iaprox,x):
    return abs(I-Iaprox(x,f))

#Function and Integral
def f(x): return x + np.sin(x)
def F(x): return x**2/2 - np.cos(x)

a,b = 0,1
x1 = np.linspace(a,b,1000)
Fx = F(x1)
I = F(b)-F(a)

rangN = (2,200)
gpoints = np.arange(*rangN)
errors = [[] for _ in range(6)]
points = [[] for _ in range(6)]
for n in gpoints:
    h = (b-a)/(n)
    x = np.arange(a,b+h,h)
    
    EabsL = (Eabs(I,rectangularL,x))
    EabsR = (Eabs(I,rectangularR,x))
    EabsM = (Eabs(I,rectangularM,x))
    EabsT = (Eabs(I,trapezoid,x))
    EabsS13 = (Eabs(I,simpson13,x))
    EabsS23 = (Eabs(I,simpson23,x))
    err = [EabsL,EabsR,EabsM,EabsT,EabsS13,EabsS23]

    #Filtering major fluctuations due to the cut of the interval at a intermediate points
    for i,e in enumerate(err):
        if i == 3 and e>(1/n**2): continue
        if i == 4 and e>(1/n**4): continue
        if i == 5 and e>(1/n**4): continue
        errors[i].append(e)
        points[i].append(n)


#Plot Settings
labels = ["Rectangular Left","Rectangular Right","Mid Point","Trapezoid","Simpson13", "Simpson23"]
for point,error,label in zip(points,errors,labels):
    plt.loglog(point, error, label = label)

n = np.linspace(*rangN)
for i in range(1,4+1):
    plt.loglog(n, 1/(n**i), ":",label=fr"1/$n^{i}$")

plt.xlabel("n");plt.ylabel("Eabs")
plt.legend(fontsize="small", bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()

# As expected, the rectangle methods have higher errors, following an 1/n dependency.
# The trapezoid method is the next one with lower errors, following an 1/n^2 dependency with the number of intervals.
# Finally, the Simpson methods present the lowest errors, following an 1/n^4 dependency.
