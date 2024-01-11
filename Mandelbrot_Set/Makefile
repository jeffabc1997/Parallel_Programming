CC = gcc
CXX = g++
LDLIBS = -lpng

CFLAGS = -lm -O3
hw2a: CFLAGS += -pthread -march=native -ffinite-math-only
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp -march=native 
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
