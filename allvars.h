/* file to store global variables*/


#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef USE_MPI
#include <mpi.h>
#ifdef USE_FFTW
#include <fftw3-mpi.h>
#endif
#endif
#ifdef ACCOMP
#include "w-stacking_omp.h"
#else
#include "w-stacking.h"
#endif
#ifdef NVIDIA
#include <cuda_runtime.h>
#endif
#define PI 3.14159265359
#define NUM_OF_SECTORS -1
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define NOVERBOSE
#define NFILES 100
#include <omp.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "numa.h"
#include <stdatomic.h>
#include <omp.h>

extern struct io
{
	FILE * pFile;
        FILE * pFile1;
        FILE * pFilereal;
        FILE * pFileimg;
} file;

extern struct ip
{
	char ufile[30];
  	char vfile[30];
  	char wfile[30];
  	char weightsfile[30];
  	char visrealfile[30];
  	char visimgfile[30];
  	char metafile[30];
        char paramfile[30];
} in;

extern struct op
{
	char outfile[30];
        char outfile1[30];
        char outfile2[30];
        char outfile3[30];
        char fftfile[30];
        char fftfile2[30];
        char fftfile3[30];
        char logfile[30];
        char extension[30];
        char timingfile[30];

} out, outparam;

extern struct meta
{

	long Nmeasures;
        long Nvis;
        long Nweights;
        long freq_per_chan;
        long polarisations;
        long Ntimes;
        double dt;
        double thours;
        long baselines;
        double uvmin;
        double uvmax;
        double wmin;
        double wmax;

} metaData;


typedef struct {
  double setup;      // time spent in initialization, init()
  double init;       // time spent in initializing arrays
  double process;    // time spent in gridding;
  double mpi;        // time spent in mpi communications
  double fftw;       //
  double kernel;     //
  double mmove;      //
  double reduce;     //
  double reduce_mpi; //
  double reduce_sh ; //
  double compose;    //
  double phase;      //
  double write;      //
  double total;
  double reduce_ring;
 } timing_t;

extern timing_t wt_timing;      // wall-clock timings
extern timing_t pr_timing;      // process CPU timing

extern struct parameter
{
        int num_threads;
        int ndatasets;
        char datapath_multi[NFILES][900];
        int grid_size_x;
        int grid_size_y;
        int num_w_planes;
        int w_support;
} param;

extern struct fileData
{
        double * uu;
        double * vv;
        double * ww;
        float * weights;
        float * visreal;
        float * visimg;
}data;


extern char filename[1000], buf[30], num_buf[30];
extern char datapath[900];
extern int xaxis, yaxis;
extern int global_rank;
extern int size;
extern int verbose_level;
extern long nsectors;
extern long startrow;
extern double resolution, dx, dw, w_supporth;

extern long * histo_send, size_of_grid;
extern double * grid, *gridss, *gridss_real, *gridss_img, *gridss_w;

#ifdef USE_MPI
    extern MPI_Comm MYMPI_COMM_WORLD;
    extern  MPI_Win slabwin;
#endif

extern long **sectorarray;


#if defined(DEBUG)
#define dprintf(LEVEL, T, t, ...) if( (verbose_level >= (LEVEL)) &&	\
				      ( ((t) ==-1 ) || ((T)==(t)) ) ) {	\
    printf(__VA_ARGS__); fflush(stdout); }

#else
#define dprintf(...)
#endif


#define CPU_TIME_wt ({ struct timespec myts; (clock_gettime( CLOCK_REALTIME, &myts ), (double)myts.tv_sec + (double)myts.tv_nsec * 1e-9);})
#define CPU_TIME_pr ({ struct timespec myts; (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &myts ), (double)myts.tv_sec + (double)myts.tv_nsec * 1e-9);})
#define CPU_TIME_th ({ struct timespec myts; (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec + (double)myts.tv_nsec * 1e-9);})


#if defined(_OPENMP)
#define TAKE_TIME_START( T ) {			\
    wt_timing.T = CPU_TIME_wt;			\
    pr_timing.T = CPU_TIME_pr; }

#define TAKE_TIME_STOP( T ) {			\
    pr_timing.T = CPU_TIME_pr - pr_timing.T;	\
    wt_timing.T = CPU_TIME_wt - wt_timing.T; }

#define TAKE_TIME( Twt, Tpr ) { Twt = CPU_TIME_wt; Tpr = CPU_TIME_pr; }
#define ADD_TIME( T, Twt, Tpr ) {		\
    pr_timing.T += CPU_TIME_pr - Tpr;		\
    wt_timing.T += CPU_TIME_wt - Twt;		\
    Tpr = CPU_TIME_pr; Twt = CPU_TIME_wt; }

#else

#define TAKE_TIME_START( T ) wt_timing.T = CPU_TIME_wt

#define TAKE_TIME_STOP( T )  wt_timing.T = CPU_TIME_wt - wt_timing.T

#define TAKE_TIME( Twt, ... ) Twt = CPU_TIME_wt;
#define ADD_TIME( T, Twt, ... ) { wt_timing.T += CPU_TIME_wt - Twt; Twt = CPU_TIME_wt;}

#endif

#define TAKE_TIMEwt_START( T) wt_timing.T = CPU_TIME_wt
#define TAKE_TIMEwt_STOP( T) wt_timing.T = CPU_TIME_wt - wt_timing.T
#define TAKE_TIMEwt( Twt ) Twt = CPU_TIME_wt;
#define ADD_TIMEwt( T, Twt ) { wt_timing.T += CPU_TIME_wt - Twt; Twt = CPU_TIME_wt; }


#if defined(__GNUC__) && !defined(__ICC) && !defined(__INTEL_COMPILER)
#define PRAGMA_IVDEP _Pragma("GCC ivdep")
#else
#define PRAGMA_IVDEP _Pragma("ivdep")
#endif

#define STRINGIFY(a) #a
#define UNROLL(N) _Pragma(STRINGIFY(unroll(N)))


#define CPU_TIME_tr ({ struct timespec myts; (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec + (double)myts.tv_nsec * 1e-9);})

#define CPU_TIME_rt ({ struct timespec myts; (clock_gettime( CLOCK_REALTIME, &myts ), (double)myts.tv_sec + (double)myts.tv_nsec * 1e-9);})

#define CPU_TIME_STAMP(t, s) { struct timespec myts; clock_gettime( CLOCK_REALTIME, &myts ); printf("STAMP t %d %s - %ld %ld\n", (t), s, myts.tv_sec, myts.tv_nsec);}

#define NSLEEP( T ) {struct timespec tsleep={0, (T)}; nanosleep(&tsleep, NULL); }


#define ACQUIRE_CTRL( ADDR, V, TIME, CMP ) {                            \
    double tstart = CPU_TIME_tr;                                        \
    atomic_thread_fence(memory_order_acquire);                          \
    int read = atomic_load((ADDR));                                     \
    while( read CMP (V) ) { NSLEEP(50);                                 \
      read = atomic_load((ADDR)); }                                     \
    (TIME) += CPU_TIME_tr - tstart; }

              
 #define ACQUIRE_CTRL_DBG( ADDR, V, CMP, TAG ) {                  \
    atomic_thread_fence(memory_order_acquire);                          \
    int read = atomic_load((ADDR));                                     \
    printf("%s %s Task %d read is %d\n", TAG, #ADDR, global_rank, read);      \
    unsigned int counter = 0;                                           \
    while( read CMP (V) ) { NSLEEP(50);                                 \
    read = atomic_load((ADDR));                                         \
    if( (++counter) % 10000 == 0 )                                      \
                        printf("%s %p Task %d read %u is %d\n",         \
                        TAG, ADDR, global_rank, counter, read);}              \
    }



#define DATA_FREE   -1
#define FINAL_FREE  -1

#define CTRL_DATA          0
#define CTRL_FINAL_STATUS  0
#define CTRL_FINAL_CONTRIB 1
#define CTRL_SHMEM_STATUS  2
#define CTRL_BLOCKS        3
#define CTRL_END           (CTRL_BLOCKS+1)

#define CTRL_BARRIER_START 0
#define CTRL_BARRIER_END   1

#define BUNCH_FOR_VECTORIZATION 128

typedef long long unsigned int int_t;
typedef struct { int Nblocks; int_t *Bstart; int_t *Bsize; } blocks_t;
extern MPI_Request *requests;
extern int thid;
extern int Ntasks_local;
extern blocks_t blocks;
extern double **swins;
extern int    **cwins;
extern int max_level;
extern double *end_4, *end_reduce;
extern int dsize_4, iter;
extern struct timing_r { double rtime, ttotal, treduce, tspin, tspin_in, tmovmemory, tsum;} timing ;
extern struct timingmpi_r{ double tmpi, tmpi_reduce, tmpi_reduce_wait, tmpi_setup;} timingmpi ; 
