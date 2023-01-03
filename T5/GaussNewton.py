"""
Class with the GaussNewton solver.

Diego Ontiveros
"""

import numpy as np


class GaussNewton:
    """
    Creates a Gaussian-Newton solver object
    """

    def __init__(self, fit_function,max_iter=1000,tol_rmse=1e-9,tol_abs = 1e-12,init_guess=None):
        """
        `fit_function` : function to fit of the form f(x,coeffs).
        `max_iter`     : maixmum number of iteration in the optimization.
        `tol`          : tolerance for RMSE step
        `inti_guess`   : Initial guess array for coeffitients
        """
        self.fit_function = fit_function
        self.max_iter = max_iter
        self.tol_rmse = tol_rmse
        self.tol_abs = tol_abs
        self.coeff = init_guess
        if init_guess is None: self.init_guess = None
        else: self.init_guess = init_guess
        
    def fit(self,x,y,init_guess=None, print_jacobian=False):
        """
        Fits coefitients by minimizing RMSE of residuals.
        `x` : Independent variable points (array)
        `y` : Response data points (array)
        `init_guess` : Initial guess for coeffitients  

        """
        self.x = np.array(x)
        self.y = np.array(y)
        if self.init_guess is None and init_guess is None: 
            raise TypeError("Initial guess array must be provided.")
        else: self.init_guess = init_guess

        self.coeff = self.init_guess
        rmse0 = np.inf

        jacobian = []
        for i in range(self.max_iter):
            print(f"\nIteration {i}: ")
            print("coeffs: ",self.coeff)
            residual = self.get_residual()                          # Gets residuals of fitting
            jacobian = self.get_jacobian()                          # Computes Jacobian of residuals/coeffs
            if print_jacobian:
                print("Jacobian:")
                print(jacobian)
            self.coeff = self.update_coeffs(residual,jacobian)      # Updates coeffitients
            rmse = np.sqrt((residual**2).sum())                     # Computes RMSE of residuals
            Eabs = abs((rmse - rmse0))

            
            print(f"RMSE residuals = {rmse}  Absolute error between iterations = {Eabs}")

            # Handling convergence by the given criteria
            if Eabs < self.tol_abs:
                print("\nRMSE difference between iterations smaller than tolerance. Converged!")
                print("Optimized coeffitients:",self.coeff)
                return self.coeff
            elif rmse < self.tol_rmse: 
                print("\nRMSE error smaller than tolerance. Converged!")
                print("Optimized coeffitients:",self.coeff)
                return self.coeff
            rmse0 = rmse
        print("\nMax number of iterations reached. NOT Converged!")

        
    def get_residual(self,coeff = None):
        """
        Returns residual error between the fitted function and data points.
        Used at each iteration with the current aproximated coeffitients.
        :::`r = phi - d`
        """

        if coeff is None: coeff = self.coeff

        yf = self.fit_function(self.x,coeff)
        return yf - self.y

    def get_jacobian(self,step=1e-6):
        """
        Returns jacobian matrix `Jij = d(ri)/d(cj)`
        """

        # Initial residuals
        r0 = self.get_residual(self.coeff)

        jacobian = []
        for i,xi in enumerate(self.coeff):          # For each coeffitient
            coeff = self.coeff.copy()               # Copy of the coeffitients
            coeff[i] += step                        # Shifts the ith coeff a given step
            r = self.get_residual(coeff = coeff)    # Computes the new residuals with the sifthed coeff
            diff = (r-r0)/step                      # Differentiation of residuals 
            jacobian.append(diff)                   # Adds diff vector to jacobian
        jacobian = np.array(jacobian).T             # Transposes to get the correct form

        return jacobian

    def update_coeffs(self,residual,jacobian):
        """
        Updates the coeffitients given the jacobian matrix and residuals array of one step.
        """
        # coeff(n+1) = coeff(n) - (J.T@J)^-1 @ J.T @ r(coeff(n))
        # self.coeff = self.coeff - np.linalg.pinv(jacobian) @ residual
        self.coeff = self.coeff - np.linalg.pinv(jacobian.T@jacobian)@jacobian.T @ residual

        return  self.coeff

    def estimate(self):
        """
        Estimated function with minimized coeffitients
        """
        return self.fit_function(self.x,self.coeff)
