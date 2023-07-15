#include <omp.h>
#include "w-stacking_omp.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef NVIDIA
#include <cuda_runtime.h>
#endif

#ifdef ACCOMP
#pragma omp  declare target
#endif
double gauss_kernel_norm(double norm, double std22, double u_dist, double v_dist)
{
     double conv_weight;
     conv_weight = norm * exp(-((u_dist*u_dist)+(v_dist*v_dist))*std22);
     return conv_weight;
}
#ifdef ACCOMP
#pragma omp end declare target
#endif



void wstack(
     int num_w_planes,
     long num_points,
     long freq_per_chan,
     long polarizations,
     double* uu,
     double* vv,
     double* ww,
     float* vis_real,
     float* vis_img,
     float* weight,
     double dx,
     double dw,
     int w_support,
     int grid_size_x,
     int grid_size_y,
     double* grid,
     int num_threads)
{
    //long index;
    long visindex;

    // initialize the convolution kernel
    // gaussian:
    int KernelLen = (w_support-1)/2;
    double std = 1.0;
    double std22 = 1.0/(2.0*std*std);
    double norm = std22/PI;

    // Loop over visibilities.
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

#ifdef ACCOMP
    long Nvis = num_points*freq_per_chan*polarizations;
    long gpu_weight_dim = Nvis/freq_per_chan;
    long gpu_grid_dim = 2*num_w_planes*grid_size_x*grid_size_y;
#pragma omp target teams distribute parallel for private(visindex) \
map(to:num_points, KernelLen, std,  std22, norm, num_w_planes, \
  uu[0:num_points], vv[0:num_points], ww[0:num_points], \
  vis_real[0:Nvis], vis_img[0:Nvis], weight[0:gpu_weight_dim], \
  grid_size_x, grid_size_y, freq_per_chan, polarizations, dx,dw, w_support, num_threads) \
  map(tofrom: grid[0:gpu_grid_dim])
#endif
    for (long i = 0; i < num_points; i++)
    {
#ifdef _OPENMP
	//int tid;
	//tid = omp_get_thread_num();
	//printf("%d\n",tid);
#endif

        visindex = i*freq_per_chan*polarizations;

        double sum = 0.0;
        int j, k;
	//if (i%1000 == 0)printf("%ld\n",i);

        /* Convert UV coordinates to grid coordinates. */
        double pos_u = uu[i] / dx;
        double pos_v = vv[i] / dx;
        double ww_i  = ww[i] / dw;
        int grid_w = (int)ww_i;
        int grid_u = (int)pos_u;
        int grid_v = (int)pos_v;

	// check the boundaries
	long jmin = (grid_u > KernelLen - 1) ? grid_u - KernelLen : 0;
	long jmax = (grid_u < grid_size_x - KernelLen) ? grid_u + KernelLen : grid_size_x - 1;
	long kmin = (grid_v > KernelLen - 1) ? grid_v - KernelLen : 0;
	long kmax = (grid_v < grid_size_y - KernelLen) ? grid_v + KernelLen : grid_size_y - 1;
        //printf("%d, %ld, %ld, %d, %ld, %ld\n",grid_u,jmin,jmax,grid_v,kmin,kmax);


        // Convolve this point onto the grid.
        for (k = kmin; k <= kmax; k++)
        {

            double v_dist = (double)k+0.5 - pos_v;

            for (j = jmin; j <= jmax; j++)
            {
                double u_dist = (double)j+0.5 - pos_u;
		long iKer = 2 * (j + k*grid_size_x + grid_w*grid_size_x*grid_size_y);


		double conv_weight = gauss_kernel_norm(norm,std22,u_dist,v_dist);
		// Loops over frequencies and polarizations
		double add_term_real = 0.0;
		double add_term_img = 0.0;
		long ifine = visindex;
		// DAV: the following two loops are performend by each thread separately: no problems of race conditions
		for (long ifreq=0; ifreq<freq_per_chan; ifreq++)
		{
		   long iweight = visindex/freq_per_chan;
	           for (long ipol=0; ipol<polarizations; ipol++)
	           {
                      if (!isnan(vis_real[ifine]))
                      {
		         //printf("%f %ld\n",weight[iweight],iweight);
                         add_term_real += weight[iweight] * vis_real[ifine] * conv_weight;
		         add_term_img += weight[iweight] * vis_img[ifine] * conv_weight;
			 //if(vis_img[ifine]>1e10 || vis_img[ifine]<-1e10)printf("%f %f %f %f %ld %ld\n",vis_real[ifine],vis_img[ifine],weight[iweight],conv_weight,ifine,num_points*freq_per_chan*polarizations);
		      }
		      ifine++;
		      iweight++;
		   }
	        }
		// DAV: this is the critical call in terms of correctness of the results and of performance
		#pragma omp atomic
		grid[iKer] += add_term_real;
		#pragma omp atomic
		grid[iKer+1] += add_term_img;
            }
        }

    }

    //for (int i=0; i<100000; i++)printf("%f\n",grid[i]);
}


#ifdef NVIDIA
#define CUDAErrorCheck(funcall)                                         \
do {                                                                    \
  cudaError_t ierr = funcall;                                           \
  if (cudaSuccess != ierr) {                                            \
    fprintf(stderr, "%s(line %d) : CUDA RT API error : %s(%d) -> %s\n", \
    __FILE__, __LINE__, #funcall, ierr, cudaGetErrorString(ierr));      \
    exit(ierr);                                                         \
  }                                                                     \
} while (0)

static inline int _corePerSM(int major, int minor)
/**
 * @brief Give the number of CUDA cores per streaming multiprocessor (SM).
 *
 * The number of CUDA cores per SM is determined by the compute capability.
 *
 * @param major Major revision number of the compute capability.
 * @param minor Minor revision number of the compute capability.
 *
 * @return The number of CUDA cores per SM.
 */
{
  if (1 == major) {
    if (0 == minor || 1 == minor || 2 == minor || 3 == minor) return 8;
  }
  if (2 == major) {
    if (0 == minor) return 32;
    if (1 == minor) return 48;
  }
  if (3 == major) {
    if (0 == minor || 5 == minor || 7 == minor) return 192;
  }
  if (5 == major) {
    if (0 == minor || 2 == minor) return 128;
  }
  if (6 == major) {
    if (0 == minor) return 64;
    if (1 == minor || 2 == minor) return 128;
  }
  if (7 == major) {
    if (0 == minor || 2 == minor || 5 == minor) return 64;
  }
  return -1;
}

void getGPUInfo(int iaccel)
{
  int corePerSM;

 struct cudaDeviceProp dev;

  CUDAErrorCheck(cudaSetDevice(iaccel));
  CUDAErrorCheck(cudaGetDeviceProperties(&dev, iaccel));
  corePerSM = _corePerSM(dev.major, dev.minor);

  printf("\n");
  printf("============================================================\n");
  printf("CUDA Device name : \"%s\"\n", dev.name);
  printf("------------------------------------------------------------\n");
  printf("Comp. Capability : %d.%d\n", dev.major, dev.minor);
  printf("max clock rate   : %.0f MHz\n", dev.clockRate * 1.e-3f);
  printf("number of SMs    : %d\n", dev.multiProcessorCount);
  printf("cores  /  SM     : %d\n", corePerSM);
  printf("# of CUDA cores  : %d\n", corePerSM * dev.multiProcessorCount);
  printf("------------------------------------------------------------\n");
  printf("global memory    : %5.0f MBytes\n", dev.totalGlobalMem / 1048576.0f);
  printf("shared mem. / SM : %5.1f KBytes\n", dev.sharedMemPerMultiprocessor / 1024.0f);
  printf("32-bit reg. / SM : %d\n", dev.regsPerMultiprocessor);
  printf("------------------------------------------------------------\n");
  printf("max # of threads / SM    : %d\n", dev.maxThreadsPerMultiProcessor);
  printf("max # of threads / block : %d\n", dev.maxThreadsPerBlock);
  printf("max dim. of block        : (%d, %d, %d)\n",
      dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
  printf("max dim. of grid         : (%d, %d, %d)\n",
      dev.maxGridSize[0],   dev.maxGridSize[1],   dev.maxGridSize[2]);
  printf("warp size                : %d\n", dev.warpSize);
  printf("============================================================\n");

  int z = 0, x = 2;
  #pragma omp target map(to:x) map(tofrom:z)
  {
    z=x+100;
  }
}

#endif






int test(int nnn)
{
	int mmm;

	mmm = nnn+1;
	return mmm;
}
