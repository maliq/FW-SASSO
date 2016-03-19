#!/usr/bin/python
import sys
import re

def process_span(string_vector):
	tuplas = []
	span_vector = re.split(r',(?=[^\]]*(?:\[|$))', string_vector[1:-1])
	try:
		for i in span_vector:

			if "-" in i: #element span
				ii = int(i.split(":")[0].split("-")[0])
				ii_end = int(i.split(":")[0].split("-")[1])+1
				for index, value in zip(range(ii, ii_end), eval(i.split(":")[1])):
					tuplas.append("%s:%s" % (index, value))
			else: #element sparse
				tuplas.append(i)
	except:
		print "exception with line:\n %s" % string_vector
	return " ".join(tuplas)

f = open(sys.argv[1],'r')
fo = open(sys.argv[2],'w')

lines = f.readlines()
counter = 0
regexp = re.compile(r'([^-\d]\d+)-(\d+[^-\d])')

type = 0 #dense 0, sparse 1, span 2 
for line in lines:
	if line:
		if counter == 0:
			if ":" in line or "[]" in line:
				type = 1
			if regexp.search(line) is not None:
				type = 2
			elements = line.split()
			bias = float(elements[0])
			fo.write("BIAS : "+str(bias)+"\n")
		elements = line.split()
		if type == 0:
			fo.write(elements[1]+" "+','.join(elements[2].strip('[]').split(',')[1:])+"\n")
		elif type == 1:
			fo.write(elements[1]+" "+elements[2].strip('[]').replace(","," ")+"\n")
		elif type == 2:
			fo.write(elements[1]+" "+process_span(elements[2])+"\n")
		counter+=1
fo.close()
f.close()
