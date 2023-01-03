###Exercici 3.1.5

#Defining dictionaries for the conversion to base >10 numbers that use lower and upper case letters
symbols_down36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"                           #bases up to 36 use upper case letters first
symbols_up36 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"   #basesfrom 37 use lower case letters first

change_down36 = {i:e for i,e in enumerate(symbols_down36)}  #Dictionaries for better data mapping
change_up36 = {i:e for i,e in enumerate(symbols_up36)}

def changeBase_letters(x,n):
    """Takes base-10 number (x) and changes it to base n. (Using the usual letter notation)"""
    new_x = ""
    quotient = x
    while quotient != 0:                    #Loop that gets the quiotient and reminder
        remainder = quotient%n
        quotient = quotient//n              #The reminder sets the digits of the new number.
        if n <= 36: new_x += str(change_down36[remainder])
        if n > 36: new_x += str(change_up36[remainder])
    
    return new_x[::-1]                      #Returns the reversed string (numbers in the right order)


#Main program
number = 3737           #Base-10 number
bases = [2,4,8,16,60]   #Bases to change the Base-10 number
for base in bases:
    new_number = changeBase_letters(number,base)
    print(f"{number} in base {base} is {new_number}")