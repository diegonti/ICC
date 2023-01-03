"""
Problem 1 - Algebra Optimization
Using Conjugate Gradient method to minimize a function.

This is just a different version of the other file but changing 
the order of updating the optimization vectors and parameters.

It was my first attempt and the optimization converges at the same value
but taking more steps. I thought it was interesting seeng that using a different
order, the same results could be attained.

Diego Ontiveros
"""

import numpy as np
import matplotlib.pyplot as plt

class Optimize:
    """ Conjugate Gradient Method class."""

    def __init__(self,function,G,b,initial_x = None):
        """
        Creates solver object for Conjugate Gradient Optimization.

        Parameters
        -----------------
        `function` : desied cuadratic function to fit of the form `0.5 * x.T @ G @ x + b.T @ x.`
        `G` :  Hessian, square and symmetric matrix. 
        `b` : Intercept or initial value array.
        `initial_x` : (optional) Array with the initial guess vector. 0 vector chosen if None.
        """
        
        # Function parameters
        self.f = function
        self.G = G
        self.b = b
        self.dim = len(b)

        if initial_x is None: self.x = np.zeros(self.dim)
        else: self.x = initial_x

        # Optimization parameters
        self.gi = self.gradient(x,G,b)
        self.si = -self.gi
        self.f_points = None


    def optimize(self,max_iter = 1000,tol_abs=1e-8,steps_info=False):
        """ Optimizes using conjugate gradient method."""
        
        flag = False
        self.f_points = []
        for i in range(max_iter):
            fi,alphai,betai = self.update(self.x,self.gi,self.si)
            if i > 0: Eabs = abs(fi - self.f_points[-1]); flag=True
            self.f_points.append(fi)

            if steps_info:
                print(f"\nIteration {i}")
                print("General point:  x =",self.x)
                print("Gradient:       g =",self.gi)
                print("s vector:       s =",self.si)
                print("beta:           b = ",betai)
                print("alpha:          a = ",alphai)
                print("Function value: f =",fi)


            if flag:
                if (Eabs < tol_abs):
                    print("\nAbsolute error between iterations lower than tolerance. Converged!")
                    print("Optimized Function value: f =",fi,"\n")
                    return self.f_points, fi
            

        print("\nMaximum number of iterations reached. NOT Converged!")
        print("Last Function value: f =",fi,"\n")
        return self.f_points, fi
    

    def update(self,x,g0,si):

        gi = self.gradient(self.G,x,self.b)
        betai = self.beta(g0,gi)
        si = self.s(si,gi,betai)
        alphai = self.alpha(si,gi,self.G)

        x += alphai*si

        fi = self.f(x,G,b)

        self.gi = gi
        self.si = si
        self.x = x

        return fi,alphai,betai

    # Functions for computing the arrays (g,s) and parameters (beta,alpha)
    # involved in the optimization steps
    def gradient(self,x,G,b): return G@x + b

    def beta(self,g0,g1): return (g1.T@g1)/(g0.T@g0)

    def s(self,s,g,beta): return -g + beta*s

    def alpha(self,s,g,G): return g.T@g/(s.T@G@s)

        
def tridiagonal(dim,k_diagonal,k_below,k_above): 
    """Creates tridiagonal matrix with specified values."""
    a = np.ones((1, dim))[0]*k_diagonal
    b = np.ones((1, dim-1))[0]*k_below
    c = np.ones((1, dim-1))[0]*k_above
    m = np.diag(a, 0) + np.diag(b, -1) + np.diag(c, 1)
    return m

def f(x,G,b):
    """Quadratic funciton to optimize."""
    return 0.5*x.T@G@x + b.T@x


########### MAIN PROGRAM ############

G = tridiagonal(4,2,-1,-1)
b = np.array([-1,0,2,np.sqrt(5)])
x = np.zeros(4)

# Random initial values (just to play a little)
# dim = 6
# a = np.random.rand(dim, dim)
# G = np.tril(a) + np.tril(a, -1).T
# b = np.random.rand(dim)
# x = np.random.rand(dim)

solver = Optimize(f,G,b,initial_x=x)
f_points, minf = solver.optimize(tol_abs=1e-5,steps_info=True)

plt.axhline(minf,c="k",ls=":")
plt.plot(f_points,"ro--")

plt.xlabel("Iteration");plt.ylabel("Function")
plt.show()


# The 
