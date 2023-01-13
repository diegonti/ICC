"""
Problem 2.1 - Numerical data types
Testing the arithmetic between different data types.
Diego Ontiveros
"""

def case1(b,c):
    """Case 1: a,b,c=int"""
    b,c= int(b),int(c)
    a = int(b/c)
    print(f"Case 1: When a,b,c = int, a = {a}")

def case2(b,c):
    """Case 2: a,b,c=float"""
    b,c = float(b),float(c)
    a = float(b/c)
    print(f"Case 2: When a,b,c = float, a = {a}")

def case3(b,c):
    """Case 3: a=float,b,c=int"""
    b,c = int(b),int(c)
    a = float(b/c)
    print(f"Case 3: When a = float and b,c = int, a = {a}")

# Main Program
b,c = 1,2
case1(b,c) # Same as b//c
case2(b,c) # Same as b/c
case3(b,c)

# You can see how unlike Fortran, in Python, even if two numbers are integers
# when dividing between them, the result becomes rational (unless it is forced with the int() function).
# It is important to keep that in mind because otherwise, there may be truncations where we don't want, or the other way around.
# In this sense, Python is more "human" since it works more like someone with a calculator, 
# on the other hand, in Fortran, when declaring the variable type, is more faithful to that declaration, so int/int gives int, not real.

# In both languages, putting a dot after the number makes it float (or real), giving it more precision and avoiding precision errors.

