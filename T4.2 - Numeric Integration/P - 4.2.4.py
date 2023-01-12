"""
Problem 4.2.4 - Integration Types: Gauss-Legandre.
Gauss-Legandre and Simpson Numerical integration of a function.
Diego Ontiveros
"""
import numpy as np
import matplotlib.pyplot as plt 

# Dictionary with the Gauss-Legandre tabulated parameters for wi and ti
parameters = {
    "n1":[[0],[2]],
    "n2": [[np.sqrt(1/3),-np.sqrt(1/3)],[1,1]],
    "n3":[[0,np.sqrt(3/5),-np.sqrt(3/5)],[8/9,5/9,5/9]],
    "n4": [[-0.33998104,0.33998104, -0.86113631,0.86113631],
            [0.65214515,0.65214515, 0.34785485,0.34785485 ]],
    "n5": [[0,0.53846931,-0.53846931,0.9061798459,-0.9061798459],
            [0.56888888,0.478628670,0.478628670,0.236926885,0.236926885]],
    "n6":[[0.66120938,-0.66120938,0.238619186,-0.238619186,0.932469514,-0.932469514],
            [0.36076157,0.36076157,0.4679139,0.4679139,0.17132449,0.17132449]],
    "n7":[[0,0.40584515,-0.4058451,0.74153118,-0.74153118,0.9491079,-0.9491079],
            [0.41795918,0.38183005,0.38183005,0.27970539,0.27970539,0.129484966,0.129484966]],
    "n8":[[0.18343464,-0.18343464,0.52553240,-0.52553240,0.79666647,-0.79666647,0.9602898,-0.9602898],
            [0.36268378,0.36268378,0.3137066,0.3137066,0.222381,0.222381,0.1012285,0.1012285]],
    "n9":[[0,0.836031107,-0.836031107,0.9681602,-0.9681602,0.32425342,-0.32425342,0.61337143,-0.61337143],
            [0.330239355,0.18064816,0.18064816,0.08127438,0.08127438,0.31234707,0.31234707,0.26061069,0.26061069]],
    "n10": [[-0.14887434,0.14887434, -0.43339539,0.43339539,0.6794095,-0.6794095,0.8650633,-0.8650633,0.97390652,-0.97390652],
            [0.29552422,0.29552422,0.269266719,0.269266719,0.21908636,0.21908636,0.149451349,0.149451349,0.06667134,0.06667134]]
}

def gaussLegendre(n,f):
    """Retunrs Gauss-Legendre Quadrature of a function f for a given degree n."""
    suma = 0
    for i in range(n):
        ti = parameters[f"n{n}"][0][i]
        wi = parameters[f"n{n}"][1][i]
        xi = (a+b)/2 + (b-a)*ti/2
        suma += wi*f(xi)
    return (b-a)/2*suma

def simpson13(x,f):
    h = abs(x[1]-x[0])
    points = f(x)
    return (h/3)*(points[0] + 2*sum(points[2:-2:2]) + 4*sum(points[1:-1:2]) + points[-1])


#################### Main Program #####################3
def f(x): return np.sin(x)  # Function to integrate
def F(x): return -np.cos(x) # Analytic integral

a,b = 0,np.pi/2             # Integration interval
I = F(b)-F(a)               # 0.999 ~ 1

# Calculates the absolute error as function of grid points (n)
EabsGL = np.array([])
EabsS = np.array([])
nPoints = np.arange(1,10+1,1)
for n in nPoints: 
    h = (b-a)/(n)
    x = np.arange(a,b+h,h)
    
    iGL = gaussLegendre(n,f)    # Gauss-Legandre integral
    iS = simpson13(x,f)         # Simpson1/3 Integral
    errGL = abs(I- iGL)         # Gauss-Legandre error
    errS = abs(I - iS)          # Simpson1/3 error

    EabsGL = np.append(EabsGL,errGL)
    EabsS = np.append(EabsS,errS)

    print(f"n = {n}:  Gauss-Legandre = {iGL}    Eabs = {errGL}")
    print(f"n = {n}:  Simpson 1/3    = {iS}    Eabs = {errS}")


# Plot Settings
plt.semilogy(nPoints, EabsGL,"x-", label="Gauss-Legandre")
plt.semilogy(nPoints, EabsS,"x-", label="Simpson 1/3")
plt.xlabel("n");plt.ylabel("Eabs")
plt.title("Gauss-Legendre quadrature absolute error for sin(x)")
plt.legend()
plt.show()

# With a much lower number of points (n) the Gauss-Legendre method returns
# lower absolute errors.