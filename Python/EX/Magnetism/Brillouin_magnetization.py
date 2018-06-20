#!/usr/bin/env python
import numpy as np
import sys
import time

start = time.time()
time.sleep(1)

from external_classes import Brillouin

#-------------------------- computation parameters -----------------------------------------------------------------
T=300.					# temperature [Kelvin] 
# Field parameters	 
Field = np.arange(0.,5.,0.2) 		# list with the field values (initial, final, step) units [Tesla]
# Brillouin function 
Spin_tot = 1./2.

#-------------------------------------------------------------------------------------------------------------------

Field_points = len(Field)
print(' ' )
print("Field points", len(Field))

if len(sys.argv)>1:
	fn = sys.argv[1]
else:
	fn = 'outputfile.txt'

f = open(fn, 'w')
f.write("# Computation parameters\n")
f.write("# Temperature:\t\t%.3f\tKelvin\n"%(T))
f.write("# Brillouin function S =\t%.1f\n"%(Spin_tot))
f.write("# \n")
f.write("# Field\t\t Brillouin(S_tot)\n") 

mu = (0.927400968/1.38064852)		# magnetic moment of an electron in Kelvin/Tesla (assuming s=1/2 and g=2)

for j in range(0,Field_points):

    B = Brillouin()
    x = Field[j]/T
    f.write( "%.5f\t\t%.5f\n" % (Field[j],B.function(Spin_tot,x)) )


f.close()
print("Output saved in",fn)


