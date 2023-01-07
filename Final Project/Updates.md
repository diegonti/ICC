## Features & Updates

List of possible features and updates to the code.


- Classes: General outer Particle() class with inner atoms He() Li(). Particle() having the general attributes, while the inner the variables (label, ...)

- Optimize: Find a way to optimize the code, specially potential calculations (exp, loops) and get distances/energies functions. Maybe using f2py for the potential functions (numba does not work well).

- System: Use System class method to writeXYZ, remove get_configuration() if not used

- .gitignore: uncheck .xyz and .pdf