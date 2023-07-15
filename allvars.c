#include "allvars.h"

struct io file;

struct ip in;

struct op out, outparam;

struct meta metaData;
timing_t wt_timing = {0};
timing_t pr_timing = {0};


struct parameter param;
struct fileData data;

char filename[1000], buf[30], num_buf[30];
char datapath[900];
int xaxis, yaxis;
int global_rank;
int size;
int verbose_level = 0;
long nsectors;
long startrow;
double resolution, dx, dw, w_supporth;

clock_t start, end, start0, startk, endk;
struct timespec begin, finish, begin0, begink, finishk;

long * histo_send, size_of_grid;
double * grid, *gridss, *gridss_real, *gridss_img, *gridss_w;

#ifdef USE_MPI
      MPI_Comm MYMPI_COMM_WORLD;
      MPI_Win slabwin;
#endif

long **sectorarray;

blocks_t blocks;
MPI_Request *requests;
int thid;
int Ntasks_local;

double **swins = NULL;
int    **cwins = NULL;
int max_level = 0;
double *end_4, *end_reduce;
int dsize_4, iter=0;  
struct timing_r timing;
struct timingmpi_r timingmpi;
