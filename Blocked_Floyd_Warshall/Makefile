NVFLAGS  := -O3 -std=c++11 -Xptxas=-v -arch=sm_61 
LDFLAGS  := -lm
CXXFLAGS := -fopenmp -O3 -lm
EXES     := hw3-1 hw3-2 hw3-3
MPFLAGS  := -Xcompiler="-fopenmp"

alls: $(EXES)

clean:
	rm -f $(EXES)


hw3-1: hw3-1.cc
	g++ $(CXXFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?
hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(MPFLAGS) $(LDFLAGS)  -o $@ $?

