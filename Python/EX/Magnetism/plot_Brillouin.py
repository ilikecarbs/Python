import pylab as plt
import sys

if len(sys.argv)>1:
	fn = sys.argv[1]
else:
	fn = 'outputfile.txt'

data = plt.loadtxt(fn,skiprows=4)

x = data[:,0]
y = data[:,1]

color = ['cyan', 'red', 'blue', 'magenta', 'green','black']
line = 0

plt.plot(x,y,'-',color=color[(line+1)%6],linewidth=2,label='Brillouin')		# Brillouin function for S_tot

# details of the plot 
points=len(x)
print(points)
xMin = 0 #x[0] - 0.5
xMax = x[points-1] + 0.5
yMin = 0 #y[0] - 0.5
yMax = y[points-1] + 0.5

plt.xlim([xMin, xMax])			# domain of the y axis
plt.ylim([yMin, yMax])			# domain of the y axis
plt.xlabel('Field [T]')			# label x axis 
plt.ylabel('Magnetization [B.M.]')	# label y axis 
plt.legend(loc='lower right')		# plot the legends
plt.show()
