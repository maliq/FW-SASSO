import pylab as plt
import sys

print sys.argv[1], sys.argv[2]
data1=plt.loadtxt(sys.argv[1])
data2=plt.loadtxt(sys.argv[2])
date3=plt.loadtxt(sys.argv[3])
fig = plt.figure(1)
ax = fig.add_subplot(111)
sorted(data2, key=lambda x: x[0])

ax.plot(data1[:, 0], data1[:, 1], '.b', lw=2, label='issvm_ep_0.5')
ax.plot(data2[:, 0], data2[:, 1], '.g', lw=2, label='issvm')
ax.plot(date3[:, 0], date3[:, 1], '.r', lw=2, label='sasso syntB')

plt.xlabel("Support Size")
plt.xscale('log')
plt.ylabel("Test Error")
plt.legend(loc='upper right');
plt.title("Test Error vs Support Size on %s"% sys.argv[4])
plt.savefig("%s_issvm_vs_sasso." % sys.argv[4])
plt.show()
