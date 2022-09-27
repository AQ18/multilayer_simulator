import lumapi

# If there's an error here, you have probably forgotten to add lumapi to the system path - see the README.

print("lumapi imported!")

import lumopt

# If there's an error here, for some reason lumopt wasn't added to the path at the same time as lumapi, despite being in a child directory. Beats me.

print("lumopt imported!")

fdtd = lumapi.FDTD()

# The most likely reason to break here is a licensing issue when initializing Lumerical.

print("FDTD object created!")

# import function defined in script format string
fdtd .eval("function helloWorld() { return \"hello world\"; }\nfunction returnFloat() { return 1.; }\nfunction addTest(a, b){ return a*b; }")
print(fdtd .helloWorld())

# It really shouldn't break here