import pylab as plt
import sys

print sys.argv[1], sys.argv[2]
sasso_data=plt.loadtxt(sys.argv[1])
issvm_data=plt.loadtxt(sys.argv[2])
sasso_syntB_data=plt.loadtxt(sys.argv[3])
fig = plt.figure(1)
ax = fig.add_subplot(111)
sorted(issvm_data, key=lambda x: x[0])

ax.plot(issvm_data[:, 0], issvm_data[:, 1], '.g', lw=2, label='issvm')
ax.plot(sasso_data[:, 0], sasso_data[:, 1], '.b', lw=2, label='sasso')
ax.plot(sasso_syntB_data[:, 0], sasso_syntB_data[:, 1], '.r', lw=2, label='sasso syntB')

plt.xlabel("Support Size")
plt.xscale('log')
plt.ylabel("Test Error")
plt.legend(loc='upper right');
plt.title("Test Error vs Support Size on %s"% sys.argv[4])
plt.savefig("%s_issvm_vs_sasso." % sys.argv[4])
plt.show()
