import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.pyplot import cm 
import sys


data = np.loadtxt(sys.argv[1]);
data2 = np.loadtxt(sys.argv[2]);

coly_without_syn = data[:,1]
colx_without_syn = data[:,0];

coly_syn = data2[:,1]
colx_syn = data2[:,0];

fig = plt.figure()
color=iter(cm.rainbow(np.linspace(0,1,3)))

ax = fig.add_subplot(1,1,1)
c=next(color)
ax.plot(colx_without_syn,coly_without_syn,c=c,label='Without Synthonization');
c=next(color)
ax.plot(colx_syn,coly_syn,c=c,label='Synthonization');

ax.set_xscale('log')
#ax.set_title("Fraction Times Bias is Greather than SV Activations")
plt.legend(loc=1, borderaxespad=0.,fontsize=10)

plt.show()

