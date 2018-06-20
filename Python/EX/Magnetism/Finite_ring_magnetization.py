#!/usr/bin/env python
import numpy as np
import sys
import time
import datetime

start = time.time()
time.sleep(1)

from external_classes import hamiltonian, Brillouin

#-------------------------- computation parameters -----------------------------------------------------------------

size = 7				# number of spins 1/2 in the ring
J_exch=-100.				# Exchange interaction [Kelvin] 
T=20.					# temperature [Kelvin] 

# Field parameters	 
Field = np.arange(0.,800,10.) 		# list with the field values (initial, final, step) units [Tesla]
theta = np.pi/2.			# angle between the field and the z axis 
delta_H = 0.01				# increment to compute numerical derivative

# Brillouin function 
Spin_tot = 0.5*size

#-------------------------------------------------------------------------------------------------------------------



# compute the eigenvalues in zero applied field 

Hx=0.
Hz=0.
H = hamiltonian(size,J_exch,J_exch,J_exch,Hx,Hz)

print(' ') 
#print "ordered eigenvalues"	
#for j in range(0,1<<size):
#	print '%d \t %.5f ' % (j,H.eigenvalue.real[j])	


if len(sys.argv)>1:
	if sys.argv[1] == 'EV':
		print("compute ONLY the eigenvalues in zero field")
		stop = time.time()
		print('time [s] = ',stop-start)
		sys.exit();

Field_points = len(Field)
print(' ') 
print("Field points"), len(Field)
Mag_gs = np.zeros(len(Field))
Mag = np.zeros(len(Field))
energy_gs = np.zeros(2)
Free_energy = np.zeros(2)

if len(sys.argv)>1:
	fn = sys.argv[1]
else:
	fn =('outputfile.txt')

f = open(fn, 'w')
f.write("# Computation parameters\n")
f.write("# Number of spins:\t\t%d\n"%(size))
f.write("# Exchange coupling:\t%.3f\t Kelvin\n"%(J_exch))
f.write("# Temperature:\t\t%.3f\tKelvin\n"%(T))
f.write("# Brillouin function S =\t%.1f\n"%(Spin_tot))
f.write("# \n")
f.write("# Field\t\t M ground state \t\t M(T) \t\t Brillouin(S_tot)\t\t Brillouin(S=1/2)\n") 

mu = (0.927400968/1.38064852)		# magnetic moment of an electron in Kelvin/Tesla (assuming s=1/2 and g=2)

for j in range(0,Field_points):

    H_eff = Field[j]
    for ih in range(0,2):
        Hx = np.sin(theta)*H_eff
        Hz = np.cos(theta)*H_eff
        H = hamiltonian(size,J_exch,J_exch,J_exch,Hx,Hz)
        energy_gs[ih] = H.eigenvalue.real[0]
        H_eff = Field[j] + delta_H
        
        zeta = 0.	
        for m in range(0,1<<size):
            zeta+= np.exp(-H.eigenvalue.real[m]/T)

            Free_energy[ih] = - T*np.log(zeta)

    Mag_gs[j] = energy_gs[0] - energy_gs[1]
    Mag_gs[j] /= (delta_H*mu)

    Mag[j] = Free_energy[0] - Free_energy[1]
    Mag[j] /=(delta_H*mu) 

    B = Brillouin()
    x = Field[j]/T
    f.write( "%.5f\t\t%.8f\t\t%.8f\t\t%.5f\t\t%.5f\n" 
    % (Field[j],Mag_gs[j],Mag[j],B.function(Spin_tot,x),B.function(0.5,x)) )


f.close()
print("Output saved in",fn)


