import matplotlib.pyplot as plt
import numpy as np 
import sys
from matplotlib.pyplot import cm 
from numpy import genfromtxt

print sys.argv[1], sys.argv[2]
#data1=np.loadtxt(sys.argv[1])
data1=genfromtxt(sys.argv[1], delimiter=';')
data2=genfromtxt(sys.argv[2], delimiter=';')


fig = plt.figure(1)
ax = fig.add_subplot(111)

data1 = np.asarray(sorted(data1, key=lambda x: x[1]))
data2 = np.asarray(sorted(data2, key=lambda x: x[1]))

color=iter(cm.rainbow(np.linspace(0,1,5)))
c=next(color)
ax.plot(data1[:, 1], data1[:, 2], c=c, lw=2, label='issvm_EP-INF_TOL-1e-5 (BASIC)')
c=next(color)
ax.plot(data2[:, 1], data2[:, 2], c=c, lw=2, label='issvm_TOL-1e-5 (AGGR)')

plt.xlabel("Support Size")
plt.xscale('log')
plt.ylabel("Test Error")
plt.legend(loc='lower left');
plt.title("Test Error vs Support Size on %s"% sys.argv[3])
plt.savefig("%s_issvm_basic_vs_aggr.png" % sys.argv[3])
plt.show()

		