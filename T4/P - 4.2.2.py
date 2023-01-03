### Exercise 4.2.1 and 4.2.2
import numpy as np
import matplotlib.pyplot as plt

#Definition of the integration functions (array-like operations)
def trapezoid(x,f):
    h = abs(x[1]-x[0])
    points = f(x)
    return h*(sum((points[1:-1]))+(points[0]+points[-1])/2)

def simpson13(x,f):
    h = abs(x[1]-x[0])
    points = f(x)
    return (h/3)*(points[0] + 2*sum(points[2:-2:2]) + 4*sum(points[1:-1:2]) + points[-1])

def simpson23(x,f):
    h = abs(x[1]-x[0])
    points = f(x)
    return (3*h/8)*(sum(points[0:-3:3]) + 3*sum(points[1:-2:3]) + 3*sum(points[2:-1:3]) + sum(points[3::3]))

#Richardson Extrapolation
def richardson(integral,coef1,coef2,x1,x2):
    return coef1*integral(x1,f) - coef2*integral(x2,f)

#Function and Integral
def f(x): return x + np.sin(x)
def F(x): return x**2/2 - np.cos(x)

#Range of Integration
a,b = 0,1
I = F(b)-F(a)

#Calculates the absolute error of the richardon extrapolation for each method
rangN = (3,200)
gpoints = np.arange(*rangN)
errors = [[] for _ in range(3)]
points = [[] for _ in range(3)]
for n in gpoints:
    h = (b-a)/(n)
    x1 = np.arange(a,b+h,h)     #gridpoints with n points
    x2 = np.arange(a,b+h,2*h)   #gridpoints with n/2 points
    
    EabsT = abs(I - (richardson(trapezoid,4/3,1/3,x1,x2)))
    EabsS13 = abs(I - (richardson(simpson13,16/15,1/15,x1,x2)))
    EabsS23 = abs(I - (richardson(simpson23,16/15,1/15,x1,x2)))
    err = [EabsT,EabsS13,EabsS23]

    #Filtering major fluctuations due to the cut of the interval at a intermediate points
    for i,e in enumerate(err):
        if i == 0 and e>(1/n**2): continue
        if i == 1 and e>(1/n**4): continue
        if i == 2 and e>(1/n**4): continue
        errors[i].append(e)
        points[i].append(n)

#Plot Settings
labels = ["Trapezoid","Simpson13", "Simpson23"]
for point,error,label in zip(points,errors,labels):
    plt.loglog(point, error, label = label)

n = np.linspace(*rangN)
for i in range(1,4+1):
    plt.loglog(n, 1/(n**i), ":",label=fr"1/$n^{i}$")

plt.xlabel("n");plt.ylabel("Eabs")
plt.title("Including Richarson Extrapolation")
plt.legend(fontsize="small", bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()

# Using the Richardson extrapolation, the errors are reduced with respect to
# using the normal integration, but follow the same tendencies and mantain the same error order.
# At higher number of points, the errors start to fluctuate due to the error values getting closer to macheps.