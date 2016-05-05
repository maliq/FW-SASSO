import pylab as plt
import sys

print sys.argv[1]
data=plt.loadtxt(sys.argv[1])
plt.figure(1)
plt.plot(data[:,0],data[:,1],color='g',lw=2)
plt.xlabel("Support Size")
plt.xscale('log')
plt.ylabel("Train Error")
plt.title("Train Error vs Support Size on %s"%sys.argv[2])
plt.savefig("Train-Error-vs-Support-Size-on-%s.pdf"%sys.argv[2])

plt.figure(2)
plt.plot(data[:,3],data[:,1],color='b',lw=2)
plt.xlabel("L1-norm")
plt.ylabel("Train Error")
plt.title("Train Error vs L1-norm on %s"%sys.argv[2])
plt.savefig("Train-Error-vs-L1norm-on-%s.pdf"%sys.argv[2])


plt.figure(3)
plt.plot(data[:,3],data[:,0],color='r',lw=2)
plt.xlabel("L1-norm")
plt.ylabel("Support Size")
plt.title("Support Size vs L1-norm on %s train set"%sys.argv[2])
plt.savefig("Support-Size-vs-L1norm-on-%s-train.pdf"%sys.argv[2])
