NVCC=nvcc
ARCH=-arch=sm_80
CFLAGS=-O3

all: assignment.exe

assignment.exe: assignment.cu
	$(NVCC) $(ARCH) $(CFLAGS) assignment.cu -o assignment.exe

clean:
	rm -f assignment.exe
