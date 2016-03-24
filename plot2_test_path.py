import pylab as plt
import sys

print sys.argv[1], sys.argv[2]
data1=plt.loadtxt(sys.argv[1])
data2=plt.loadtxt(sys.argv[2])
fig = plt.figure(1)
ax = fig.add_subplot(111)
sorted(data2, key=lambda x: x[0])

ax.plot(data2[:,0],data2[:,1],'.g',lw=2,label='issvm')
ax.plot(data1[:,0],data1[:,1],'xb',lw=2,label='sasso')
plt.xlabel("Support Size")
plt.xscale('log')
plt.ylabel("Test Error")
plt.legend(loc='upper right');
plt.title("Test Error vs Support Size on %s"% sys.argv[3])
plt.savefig("%s_issvm_vs_sasso." % sys.argv[3])
plt.show()
