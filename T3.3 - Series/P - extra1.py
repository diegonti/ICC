### Extra 1: Simplified fraction
"""
Problem Extra - Decimal to fraction.
Returns the simplified fraction version of a decimal.
Diego Ontiveros
"""

def findFactors(x:int):
    factors = []
    for i in range(1,int(x/2)+1):
        if x%i == 0: factors.append(i)
   

def toFraction(x):
    """Returns the simplified fraction that represents a decimal number."""

    num = str(x)
    decimals = len(num.split(".")[-1])

    numerator = int(x * 10**decimals)
    denominator = int(10**decimals)
    print(numerator, denominator)



toFraction(0.1234)
