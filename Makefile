CXX? = g++

#--- debug flags
#CFLAGS = -std=c++11 -Wall -Wconversion -O3 -fPIC -g3

#--- optimize flags
CFLAGS = -std=c++11 -Wall -Wconversion -O3 -fPIC -funroll-loops -fomit-frame-pointer -DNDEBUG
#CFLAGS = -std=c++11 -Wall -Wconversion -O3 -fPIC


FW_LIB = SASSO.o SASSO_kernel.o sCache.o mCache.o Simple_Kernel_Function.o TEST_kernel.o

all: sasso-train

sasso-train: SASSO_train.h SASSO_train.cpp $(FW_LIB)
	$(CXX) $(CFLAGS) SASSO_train.cpp $(FW_LIB) -o sasso-train -lm

Simple_Kernel_Function.o: Simple_Kernel_Function.cpp Simple_Kernel_Function.h SASSO_definitions.h
	$(CXX) $(CFLAGS) -c Simple_Kernel_Function.cpp
SASSO.o: SASSO.cpp SASSO.h SASSO_definitions.h
	$(CXX) $(CFLAGS) -c SASSO.cpp
SASSO_kernel.o: SASSO_kernel.cpp SASSO_kernel.h sCache.h mCache.h SASSO_definitions.h
	$(CXX) $(CFLAGS) -c SASSO_kernel.cpp
TEST_kernel.o: TEST_kernel.cpp TEST_kernel.h mCache.h SASSO_definitions.h
	$(CXX) $(CFLAGS) -c TEST_kernel.cpp
sCache.o: sCache.cpp sCache.h SASSO_definitions.h
	$(CXX) $(CFLAGS) -c sCache.cpp
mCache.o: mCache.cpp mCache.h SASSO_definitions.h
	$(CXX) $(CFLAGS) -c mCache.cpp
clean:
	rm -f *~ *.o sasso-train 
