# comment/uncomment the various options depending hoe you want to build the program
# Set default values for compiler options if no systype options are given or found
CC        = mpiCC
CXX       = mpiCC
OPTIMIZE  = -std=c++11 -Wall -g -O2
MPICHLIB  = -lmpich
SWITCHES =

ifdef SYSTYPE
SYSTYPE := $(SYSTYPE)
include Build/Makefile.$(SYSTYPE)
else
include Build/Makefile.systype
endif

FFTW_MPI_INC = -I/${HOME}/include
FFTW_MPI_LIB = /${HOME}/lib 

CFLAGS += $(FFTW_MPI_INC) -I/proto.h
LIBS = -L$(FFTW_MPI_LIB) -lfftw3_mpi -lfftw3 -lm #-lcudart  -lcuda

# create MPI code
OPT += -DUSE_MPI
#OPT += -DACCOMP
# use FFTW (it can be switched on ONLY if MPI is active)
ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
   OPT += -DUSE_FFTW
	 LIBS = -L$(FFTW_MPI_LIB) -lfftw3_mpi -lfftw3 -lm 
endif

#OPT += -DNVIDIA
# perform one-side communication (suggested) instead of reduce (only if MPI is active)
OPT += -DONE_SIDE
# write the full 3D cube of gridded visibilities and its FFT transform
#OPT += -DWRITE_DATA
# write the final image
#OPT += -DWRITE_IMAGE
# perform w-stacking phase correction
OPT += -DPHASE_ON
# perform ring reduce
OPT += -DRING
#perform binomial reduce
#OPT += -DBINOMIAL
#perform debuging
#OPT += -DDEBUG


DEPS = w-stacking.h main.c w-stacking.cu phase_correction.cu allvars.h init.c gridding.c fourier_transform.c result.c reduce.c numa.h
COBJ = w-stacking.o main.o phase_correction.o allvars.o init.o gridding.o fourier_transform.o result.o reduce.o numa.o

w-stacking.c: w-stacking.cu
	cp w-stacking.cu w-stacking.c

phase_correction.c: phase_correction.cu
	cp phase_correction.cu phase_correction.c

ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
%.o: %.c $(DEPS)
	$(MPICC) $(OPTIMIZE) $(OPT) -fopenmp -c -g -O0 -o $@ $< $(CFLAGS)
else
%.o: %.c $(DEPS)
	$(CC) $(OPTIMIZE) $(OPT) -fopenmp -c -g -O0 -o $@ $< $(CFLAGS)
endif

serial: $(COBJ)
	$(CC) $(OPTIMIZE) $(OPT) -o w-stackingCfftw_serial  $^ $(LIBS)

serial_omp: phase_correction.c
	$(CC)  $(OPTIMIZE) $(OPT) -o w-stackingOMP_serial main.c init.c gridding.c fourier_transform.c result.c reduce.c w-stacking_omp.c    $(CFLAGS) $(LIBS)

simple_mpi: phase_correction.c
	$(MPICC) $(OPTIMIZE) $(OPT) -o w-stackingMPI_simple w-stacking_omp.c main.c init.c gridding.c fourier_transform.c result.c reduce.c phase_correction.c  $(CFLAGS) $(LIBS)

mpi_omp: phase_correction.c
	$(MPICC) $(OPTIMIZE) $(OPT) -o w-stackingMPI_omp w-stacking_omp.c main.c init.c gridding.c fourier_transform.c result.c reduce.c phase_correction.c  $(CFLAGS) $(LIBS)

serial_cuda:
	$(NVCC) $(NVFLAGS) -c w-stacking.cu phase_correction.cu $(NVLIB)
	$(CC)  $(OPTIMIZE) $(OPT) -c main.c init.c gridding.c fourier_transform.c result.c reduce.c $(CFLAGS) $(LIBS)
	$(CXX) $(OPTIMIZE) $(OPT) -o w-stackingfftw_serial w-stacking-fftw.o w-stacking.o phase_correction.o $(CFLAGS) $(NVLIB) -lm

mpi: $(COBJ)
	$(MPICC) $(OPTIMIZE) -fopenmp -o w-stackingCfftw $^  $(CFLAGS) $(LIBS)

mpi_cuda:
	$(NVCC)   $(NVFLAGS) -c w-stacking.cu phase_correction.cu $(NVLIB)
	$(MPICC)  $(OPTIMIZE) $(OPT) -c main.c init.c fourier_transform.c result.c gridding.c reduce.c $(CFLAGS) $(LIBS)
	$(MPIC++) $(OPTIMIZE) $(OPT)   -o w-stackingfftw w-stacking-fftw.o w-stacking.o phase_correction.o $(NVLIB) $(CFLAGS) $(LIBS)

clean:
	rm *.o
	rm w-stacking.c
	rm phase_correction.c
