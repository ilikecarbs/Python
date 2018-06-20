import numpy as np
from numpy import linalg as LA

class operator(object):

    def sigma_x(self,state,site):
        return state^(1<<site)

    def sigma_z(self,state,site):
        return (((state>>site)&1)<<1) - 1

class hamiltonian(object):

    def __init__(self,n_spins=1,Jx=1.,Jy=1.,Jz=1.,Hx=0.,Hz=0.):

        mu = (0.927400968/1.38064852)		# magnetic moment of an electron in Kelvin/Tesla (assuming s=1/2 and g=2)
        
        op = operator()

        dim = 1 << n_spins  
        self.matrix = np.zeros(shape=(dim,dim))
        state = 0 
        middle_state = 0 
        new_state = 0 
        site = 0
        site_next = 0
        sigma_1 = 0
        sigma_2 = 0
        op = operator()

        for site in range(0,n_spins):
        		site_next = site + 1 
        		if (site_next == n_spins): site_next = 0 
        
        		for state in range(0,dim):
        
        #		Exchange terms 	
        			sigma_1 = op.sigma_z(state,site)
        			sigma_2 = op.sigma_z(state,site_next)
        			self.matrix[state,state] -= Jz*sigma_2*sigma_1		# sigma_z_1 sigma_z_2
        
        			middle_state = op.sigma_x(state,site)
        			new_state = op.sigma_x(middle_state,site_next)
        			self.matrix[new_state,state] -= Jx-Jy*sigma_2*sigma_1	# sigma_x_1 sigma_x_2 + sigma_y_1 sigma_y_2 
        
        #		Zeeman terms along z and along x 
        			new_state = op.sigma_x(state,site)
        			self.matrix[new_state,state] += mu*Hx 	
        			self.matrix[state,state] += mu*Hz*sigma_1
        
        	
        self.eigenvalue = np.zeros(shape=(dim))
        self.eigenvector = np.zeros(shape=(dim,dim))
        self.eigenvalue, self.eigenvector = LA.eig(self.matrix)
        
        #	order eigenvalues (achtung: if you mean to use the eigenvectors!!!)
        idx = self.eigenvalue.argsort()[::1]   # [::-1] inverted order 
        self.eigenvalue = self.eigenvalue[idx]
        self.eigenvector = self.eigenvector[:,idx]


class Brillouin(object):

    def function(self,Spin,x):

        mu = (0.927400968/1.38064852)			# Bohr magneton in Kelvin/Tesla (assuming s=1/2 and g=2)

        if ( abs(x)>1.e-8 ):
        		tmp1 = (2*Spin + 1.)/(2*Spin)
        		tmp2 = 1./(2*Spin)
        		x *= 2.*Spin*mu				# g=2 times the total spin, times the Bohr magneton (mu)	
        		fct = tmp1/np.tanh(tmp1*x)
        		fct -= tmp2/np.tanh(tmp2*x)
        else:
        #		print 'Brillouin function vanishes'
        		fct = 0.
        return fct*2.*Spin				# g=2 times the total spin
        
        
