#include<stdio.h>
#include "allvars.h"
#include "proto.h"

//Revamped by NANDHANA SAKTHIVEL as part of her Master thesis, DSSC, UNITS, Italy


void write_result()
{

  TAKE_TIME_STOP( total );
       
  if (global_rank == 0)
    {
      printf("%14s time : %f sec\n", "Setup", wt_timing.setup);
      printf("%14s time : %f sec\n", "Process", wt_timing.process);
      printf("%14s time : %f sec\n", "Reduce", wt_timing.reduce);
     #if defined(USE_MPI)
     #if defined(ONE_SIDE)
      //printf("%14s time : %f sec\n", "Reduce sh", wt_timing.reduce_sh);
      printf("%14s time : %f sec\n", "Reduce ring", wt_timing.reduce_ring);
      printf("%14s time : %f sec\n", "Shared mem reduce ring", timing.treduce);
      printf("%14s time : %f sec\n", "Shmem reduce multi host", timing.treduce+timingmpi.tmpi_reduce);
      //printf("%14s time : %f sec\n", "Mmove", wt_timing.mmove);
      //printf("%14s time : %f sec\n", "ReduceMPI", wt_timing.reduce_mpi);
     #endif
      printf("%14s time : %f sec\n", "MPI", wt_timing.mpi);
     #endif
      printf("%10s Kernel time = %f, Array Composition time %f, Reduce time: %f sec\n", "",
	     wt_timing.kernel,wt_timing.compose,wt_timing.reduce);
     #ifdef USE_FFTW
      printf("%14s time : %f sec\n", "FFTW", wt_timing.fftw);
      printf("%14s time : %f sec\n", "Phase",wt_timing.phase);
     #endif
      printf("%14s time : %f sec\n\n", "TOTAL", wt_timing.total);
      
      if(param.num_threads > 1)
	{
	  printf("%14s time : %f sec\n", "PSetup", pr_timing.setup);
	  printf("%14s time : %f sec\n", "PProcess", pr_timing.process);
	  printf("%10s PKernel time = %f, PArray Composition time %f, PReduce time: %f sec\n", "",
		 pr_timing.kernel,pr_timing.compose,pr_timing.reduce);
	 #ifdef USE_FFTW
	  printf("%14s time : %f sec\n", "PFFTW", pr_timing.fftw);
	  printf("%14s time : %f sec\n", "PPhase", pr_timing.phase);
	 #endif
	  printf("%14s time : %f sec\n", "PTOTAL", pr_timing.total);
	}
    }
  
  if (global_rank == 0)
    {
      file.pFile = fopen (out.timingfile,"w");
      if (param.num_threads == 1)
	{
	  fprintf(file.pFile, "%f %f %f %f %f %f %f\n",
		  wt_timing.setup, wt_timing.kernel, wt_timing.compose,
		  wt_timing.reduce,wt_timing.fftw,wt_timing.phase, wt_timing.total);
	} else {
	fprintf(file.pFile, "%f %f %f %f %f %f %f\n",
		pr_timing.setup, pr_timing.kernel, pr_timing.compose,
		pr_timing.reduce,pr_timing.fftw,pr_timing.phase, pr_timing.total);
      }
      fclose(file.pFile);
    }
  
 #ifdef USE_MPI
  MPI_Win_fence(0,slabwin);
  MPI_Win_free(&slabwin);
 #endif
}
