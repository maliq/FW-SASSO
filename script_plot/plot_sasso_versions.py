import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.pyplot import cm

print sys.argv[1], sys.argv[2]
data1=np.loadtxt(sys.argv[1])
data2=np.loadtxt(sys.argv[2])
data3=np.loadtxt(sys.argv[3])
data4=np.loadtxt(sys.argv[4])
fig = plt.figure(1)
ax = fig.add_subplot(111)
data1 = np.asarray(sorted(data1, key=lambda x: x[0]))
data2 = np.asarray(sorted(data2, key=lambda x: x[0]))
data3 = np.asarray(sorted(data3, key=lambda x: x[0]))
data4 = np.asarray(sorted(data4, key=lambda x: x[0]))


color=iter(cm.rainbow(np.linspace(0,1,5)))

c=next(color)
# ax.plot(data1[:, 0], data1[:, 1], c=c, lw=2, label='sasso ')
c=next(color)
ax.plot(data2[:, 0], data2[:, 1], c=c, lw=2, label='sasso syntB')
c=next(color)
# ax.plot(data3[:, 0], data3[:, 1], c=c, lw=2, label='sasso AD')
c=next(color)
ax.plot(data4[:, 0], data4[:, 1], c=c, lw=2, label='sasso AD syntB')

plt.xlabel("Support Size")
plt.xscale('log')
plt.ylabel("Test Error")
plt.legend(loc='upper right');
plt.title("Test Error vs Support Size on %s"% sys.argv[5])
plt.savefig("%s_sasso_variants." % sys.argv[5])
plt.show()
