### Exercise 4.2.1 and 4.2.2
import numpy as np
import matplotlib.pyplot as plt


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

def f(x): return x + np.sin(x)
def F(x): return x**2/2 - np.cos(x)

a,b = 0,1
x1 = np.linspace(a,b,1000)
Fx = F(x1)
I = F(b)-F(a)

# gpoints = np.array([i**3 for i in np.arange(1,7)])
rangN = (2,1000)
gpoints = np.arange(*rangN)
EabsL,EabsR,EabsM,EabsT,EabsS13,EabsS23 = [[] for _ in range(6)]
errors = [[] for _ in range(6)]
for n in gpoints:
    h = (b-a)/(n)
    x = np.arange(a,b+h,h) #Quizas poner b+h ??
   
     
    EabsL.append(Eabs(I,rectangularL,x))
    EabsR.append(Eabs(I,rectangularR,x))
    EabsM.append(Eabs(I,rectangularM,x))
    EabsT.append(Eabs(I,trapezoid,x))
    EabsS13.append(Eabs(I,simpson13,x))
    EabsS23.append(Eabs(I,simpson23,x))

# gSpoints = np.arange(2,1000,1/2)
# for n in gSpoints:
#     h = (b-a)/(n-1)
#     xs = np.arange(a,b+h,h)
#     EabsS.append(Eabs(I,simpson,xs))


errors = [EabsL,EabsR,EabsM,EabsT,EabsS13,EabsS23]
labels = ["Rectangular Left","Rectangular Right","Mid Point","Trapezoid","Simpson1/3","Simpson2/3"]
for error,label in zip(errors,labels):
    plt.loglog(gpoints, error, "x-",label = label)

n = np.linspace(*rangN)
for i in range(1,4+1):
    plt.loglog(n, 1/(n**i), ":",label=fr"1/$n^{i}$")

plt.xlabel("n");plt.ylabel("Eabs")
plt.legend(fontsize="small")
plt.show()