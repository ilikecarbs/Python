import pylab as plt
import sys

if len(sys.argv)>1:
	fn = sys.argv[1]
else:
	fn = 'outputfile.txt'

data = plt.loadtxt(fn,skiprows=7)

x = data[:,0]
y = data[:,1:5]

color = ['cyan', 'red', 'blue', 'magenta', 'green','black']
line = 0

plt.plot(x,y[:,1],'o',color=color[(line+1)%6],markersize=5,label='M(T) ring')			# magnetization at the chosen T (not zero)
plt.plot(x,y[:,0],'-',color=color[line%6],linewidth=2,label='M(T=0) ring')			# ground-state magnetization 
plt.plot(x,y[:,2],'--',color=color[(line+2)%6],linewidth=2,label='Brillouin S=N/2')		# Brillouin function for S_tot = 1/2 *(# spins)
plt.plot(x,y[:,3],'--',color=color[(line+3)%6],linewidth=2,label='Brillouin s=1/2')		# Brillouin function for s = 1/2 

# details of the plot 
xMin = data[0,0] - 0.5
points=len(x)
print(points)
xMax = data[points-1,0] + 0.5
yMin = data[0,3]- 0.5
yMax = data[points-1,3] + 0.5

plt.xlim([xMin, xMax])			# domain of the y axis
plt.ylim([yMin, yMax])			# domain of the y axis
plt.xlabel('Field [T]')			# label x axis 
plt.ylabel('Magnetization [B.M.]')	# label y axis 
plt.legend(loc='lower right')		# plot the legends
plt.show()
