import matplotlib.pyplot as plt
import numpy as np 
import sys
from matplotlib.pyplot import cm 
from numpy import genfromtxt

data1=genfromtxt(sys.argv[1], delimiter=';')
data2=genfromtxt(sys.argv[2], delimiter=';')
data3=genfromtxt(sys.argv[3], delimiter=';')
data4=genfromtxt(sys.argv[4], delimiter=';')
data5=genfromtxt(sys.argv[5], delimiter=';')


fig = plt.figure(1)
ax = fig.add_subplot(111)

data1 = np.asarray(sorted(data1, key=lambda x: x[1]))
data2 = np.asarray(sorted(data2, key=lambda x: x[1]))
data3 = np.asarray(sorted(data3, key=lambda x: x[1]))
data4 = np.asarray(sorted(data4, key=lambda x: x[1]))
data5 = np.asarray(sorted(data5, key=lambda x: x[1]))

color=iter(cm.rainbow(np.linspace(0,1,5)))

c=next(color)
ax.plot(data1[:, 1], data1[:, 2], c=c, lw=2, label='issvm_TOL-1e-2')
c=next(color)
ax.plot(data2[:, 1], data2[:, 2], c=c, lw=2, label='issvm_TOL-1e-3')
c=next(color)
ax.plot(data3[:, 1], data3[:, 2], c=c, lw=2, label='issvm_TOL-1e-4')
c=next(color)
ax.plot(data4[:, 1], data4[:, 2], c=c, lw=2, label='issvm_TOL-1e-5')
c=next(color)
ax.plot(data5[:, 1], data5[:, 2], c=c, lw=2, label='issvm_TOL-1e-6')

plt.xlabel("Support Size")
plt.xscale('log')
plt.ylabel("Test Error")
plt.legend(loc='upper right');
plt.title("Test Error vs Support Size on %s"% sys.argv[6])
plt.savefig("%s_all_TOLs.png" % sys.argv[6])
plt.show()

		