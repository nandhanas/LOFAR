#ifdef _OPENMP
#include <omp.h>
#endif
#include "w-stacking.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef ACCOMP
#pragma omp  declare target
#endif
#ifdef __CUDACC__
double __device__
#else
double
#endif
gauss_kernel_norm(double norm, double std22, double u_dist, double v_dist)
{
     double conv_weight;
     conv_weight = norm * exp(-((u_dist*u_dist)+(v_dist*v_dist))*std22);
     return conv_weight;
}
#ifdef ACCOMP
#pragma omp end declare target
#endif

#ifdef __CUDACC__
//double __device__ gauss_kernel_norm(double norm, double std22, double u_dist, double v_dist)
//{
//     double conv_weight;
//     conv_weight = norm * exp(-((u_dist*u_dist)+(v_dist*v_dist))*std22);
//     return conv_weight;
//}

__global__ void convolve_g(
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
     int KernelLen,
     int grid_size_x,
     int grid_size_y,
     double* grid,
     double std22)

{
	//printf("DENTRO AL KERNEL\n");
        long gid = blockIdx.x*blockDim.x + threadIdx.x;
	if(gid < num_points)
	{
	long i = gid;
        long visindex = i*freq_per_chan*polarizations;
	double norm = std22/PI;

        int j, k;

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
        //printf("%ld, %ld, %ld, %ld\n",jmin,jmax,kmin,kmax);


        // Convolve this point onto the grid.
        for (k = kmin; k <= kmax; k++)
        {

            double v_dist = (double)k+0.5 - pos_v;

            for (j = jmin; j <= jmax; j++)
            {
                double u_dist = (double)j+0.5 - pos_u;
                long iKer = 2 * (j + k*grid_size_x + grid_w*grid_size_x*grid_size_y);
		//printf("--> %ld %d %d %d\n",iKer,j,k,grid_w);

                double conv_weight = gauss_kernel_norm(norm,std22,u_dist,v_dist);
                // Loops over frequencies and polarizations
                double add_term_real = 0.0;
                double add_term_img = 0.0;
                long ifine = visindex;
                for (long ifreq=0; ifreq<freq_per_chan; ifreq++)
		{
		   long iweight = visindex/freq_per_chan;
                   for (long ipol=0; ipol<polarizations; ipol++)
                   {
                      double vistest = (double)vis_real[ifine];
                      if (!isnan(vistest))
                      {
                         add_term_real += weight[iweight] * vis_real[ifine] * conv_weight;
                         add_term_img += weight[iweight] * vis_img[ifine] * conv_weight;
		      }
                      ifine++;
		      iweight++;
                   }
		}
		atomicAdd(&(grid[iKer]),add_term_real);
		atomicAdd(&(grid[iKer+1]),add_term_img);
            }
        }
	}
}
#endif
#ifdef ACCOMP
#pragma  omp declare target
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
    long i;
    //long index;
    long visindex;

    // initialize the convolution kernel
    // gaussian:
    int KernelLen = (w_support-1)/2;
    double std = 1.0;
    double std22 = 1.0/(2.0*std*std);
    double norm = std22/PI;

    // Loop over visibilities.
// Switch between CUDA and GPU versions
#ifdef __CUDACC__
    // Define the CUDA set up
    int Nth = NTHREADS;
    long Nbl = (long)(num_points/Nth) + 1;
    if(NWORKERS == 1) {Nbl = 1; Nth = 1;};
    long Nvis = num_points*freq_per_chan*polarizations;
    printf("Running on GPU with %d threads and %d blocks\n",Nth,Nbl);

    // Create GPU arrays and offload them
    double * uu_g;
    double * vv_g;
    double * ww_g;
    float * vis_real_g;
    float * vis_img_g;
    float * weight_g;
    double * grid_g;

    //for (int i=0; i<100000; i++)grid[i]=23.0;
    cudaError_t mmm;
    mmm=cudaMalloc(&uu_g,num_points*sizeof(double));
    mmm=cudaMalloc(&vv_g,num_points*sizeof(double));
    mmm=cudaMalloc(&ww_g,num_points*sizeof(double));
    mmm=cudaMalloc(&vis_real_g,Nvis*sizeof(float));
    mmm=cudaMalloc(&vis_img_g,Nvis*sizeof(float));
    mmm=cudaMalloc(&weight_g,(Nvis/freq_per_chan)*sizeof(float));
    mmm=cudaMalloc(&grid_g,2*num_w_planes*grid_size_x*grid_size_y*sizeof(double));
    mmm=cudaMemset(grid_g,0.0,2*num_w_planes*grid_size_x*grid_size_y*sizeof(double));

    mmm=cudaMemcpy(uu_g, uu, num_points*sizeof(double), cudaMemcpyHostToDevice);
    mmm=cudaMemcpy(vv_g, vv, num_points*sizeof(double), cudaMemcpyHostToDevice);
    mmm=cudaMemcpy(ww_g, ww, num_points*sizeof(double), cudaMemcpyHostToDevice);
    mmm=cudaMemcpy(vis_real_g, vis_real, Nvis*sizeof(float), cudaMemcpyHostToDevice);
    mmm=cudaMemcpy(vis_img_g, vis_img, Nvis*sizeof(float), cudaMemcpyHostToDevice);
    mmm=cudaMemcpy(weight_g, weight, (Nvis/freq_per_chan)*sizeof(float), cudaMemcpyHostToDevice);

    // Call main GPU Kernel
    convolve_g <<<Nbl,Nth>>> (
	       num_w_planes,
               num_points,
               freq_per_chan,
               polarizations,
               uu_g,
               vv_g,
               ww_g,
               vis_real_g,
               vis_img_g,
               weight_g,
               dx,
               dw,
               KernelLen,
               grid_size_x,
               grid_size_y,
               grid_g,
	       std22);

    mmm = cudaMemcpy(grid, grid_g, 2*num_w_planes*grid_size_x*grid_size_y*sizeof(double), cudaMemcpyDeviceToHost);
    //for (int i=0; i<100000; i++)printf("%f\n",grid[i]);
    printf("CUDA ERROR %s\n",cudaGetErrorString(mmm));
    mmm=cudaFree(uu_g);
    mmm=cudaFree(vv_g);
    mmm=cudaFree(ww_g);
    mmm=cudaFree(vis_real_g);
    mmm=cudaFree(vis_img_g);
    mmm=cudaFree(weight_g);
    mmm=cudaFree(grid_g);

// Switch between CUDA and GPU versions
# else

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

#ifdef ACCOMP
    long Nvis = num_points*freq_per_chan*polarizations;
  //  #pragma omp target data map(to:uu[0:num_points], vv[0:num_points], ww[0:num_points], vis_real[0:Nvis], vis_img[0:Nvis], weight[0:Nvis/freq_per_chan])
  //  #pragma omp target teams distribute parallel for  map(to:uu[0:num_points], vv[0:num_points], ww[0:num_points], vis_real[0:Nvis], vis_img[0:Nvis], weight[0:Nvis/freq_per_chan]) map(tofrom: grid[0:2*num_w_planes*grid_size_x*grid_size_y])
#else
    #pragma omp parallel for private(visindex)
#endif
    for (i = 0; i < num_points; i++)
    {
#ifdef _OPENMP
	//int tid;
	//tid = omp_get_thread_num();
	//printf("%d\n",tid);
#endif

        visindex = i*freq_per_chan*polarizations;

        //double sum = 0.0;
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
// End switch between CUDA and CPU versions
#endif
    //for (int i=0; i<100000; i++)printf("%f\n",grid[i]);
}
#ifdef ACCOMP
#pragma  omp end declare target
#endif

int test(int nnn)
{
	int mmm;

	mmm = nnn+1;
	return mmm;
}
