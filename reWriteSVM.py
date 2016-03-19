import sys

f = open(sys.argv[1],'r')
fo = open(sys.argv[2],'w')

lines = f.readlines()
counter = 0
for line in lines:
	if line:
		if counter == 0:
			elements = line.split()
			bias = float(elements[0])
			fo.write("BIAS : "+str(bias)+"\n")
			fo.write(elements[1]+" "+elements[2].strip('[]').replace(","," ")+"\n")
		else:
			elements = line.split()
			fo.write(elements[1]+" "+elements[2].strip('[]').replace(","," ")+"\n")
		counter+=1
fo.close()
f.close()
