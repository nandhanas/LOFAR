#include <stdio.h>
#include "allvars.h"
#include "proto.h"

//Revamped by NANDHANA SAKTHIVEL as part of her Master thesis, DSSC, UNITS, Italy

void fftw_data()
{
  
 #ifdef USE_FFTW
  // FFT transform the data (using distributed FFTW)
  if(global_rank == 0)printf("PERFORMING FFT\n");

  TAKE_TIME_START(fftw);

  fftw_plan plan;
  fftw_complex *fftwgrid;
  ptrdiff_t alloc_local, local_n0, local_0_start;
  double norm = 1.0/(double)(param.grid_size_x*param.grid_size_y);
  
  // map the 1D array of complex visibilities to a 2D array required by FFTW (complex[*][2])
  // x is the direction of contiguous data and maps to the second parameter
  // y is the parallelized direction and corresponds to the first parameter (--> n0)
  // and perform the FFT per w plane
  alloc_local = fftw_mpi_local_size_2d(param.grid_size_y, param.grid_size_x, MPI_COMM_WORLD,&local_n0, &local_0_start);
  fftwgrid    = fftw_alloc_complex(alloc_local);
  plan        = fftw_mpi_plan_dft_2d(param.grid_size_y, param.grid_size_x, fftwgrid, fftwgrid, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
  
  long fftwindex = 0;
  long fftwindex2D = 0;
  for (int iw=0; iw<param.num_w_planes; iw++)
    {
      //printf("FFTing plan %d\n",iw);
      //select the w-plane to transform
      for (int iv=0; iv<yaxis; iv++)
		{
		  for (int iu=0; iu<xaxis; iu++)
		    {
		      fftwindex2D = iu + iv*xaxis;
		      fftwindex = 2*(fftwindex2D + iw*xaxis*yaxis);
		      fftwgrid[fftwindex2D][0] = grid[fftwindex];
		      fftwgrid[fftwindex2D][1] = grid[fftwindex+1];
		    }
		}
	      
	      // do the transform for each w-plane        
	      fftw_execute(plan);
	      
	      // save the transformed w-plane
	      for (int iv=0; iv<yaxis; iv++)
		{
		  for (int iu=0; iu<xaxis; iu++)
		    {
		      fftwindex2D = iu + iv*xaxis;
		      fftwindex = 2*(fftwindex2D + iw*xaxis*yaxis);
		      gridss[fftwindex] = norm*fftwgrid[fftwindex2D][0];
		      gridss[fftwindex+1] = norm*fftwgrid[fftwindex2D][1];
		    }
		}
	      
	    }

	  fftw_destroy_plan(plan);
	  fftw_free(fftwgrid);
	  
	 #ifdef USE_MPI
	  MPI_Win_fence(0,slabwin);
	  MPI_Barrier(MPI_COMM_WORLD);
	 #endif
	  
	  TAKE_TIME_STOP(fftw);

	 #endif
	}

	void write_fftw_data(){

	  // Write results
	  
	 #ifdef USE_FFTW
	  
	  double twt, tpr;
	    
	 #ifdef WRITE_DATA

	  TAKE_TIME(twt, tpr);
	  
	 #ifdef USE_MPI
	  MPI_Win writewin;
	  MPI_Win_create(gridss, size_of_grid*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &writewin);
	  MPI_Win_fence(0,writewin);
	 #endif
	  if (global_rank == 0)
	    {
	      printf("WRITING FFT TRANSFORMED DATA\n");
	      file.pFilereal = fopen (out.fftfile2,"wb");
	      file.pFileimg = fopen (out.fftfile3,"wb");
	     #ifdef USE_MPI
	      for (int isector=0; isector<nsectors; isector++)
		{
		  MPI_Win_lock(MPI_LOCK_SHARED,isector,0,writewin);
		  MPI_Get(gridss_w,size_of_grid,MPI_DOUBLE,isector,0,size_of_grid,MPI_DOUBLE,writewin);
		  MPI_Win_unlock(isector,writewin);
		  for (long i=0; i<size_of_grid/2; i++)
		    {
		      gridss_real[i] = gridss_w[2*i];
		      gridss_img[i] = gridss_w[2*i+1];
		    }
		  if (param.num_w_planes > 1)
		    {
		      for (int iw=0; iw<param.num_w_planes; iw++)
			for (int iv=0; iv<yaxis; iv++)
			  for (int iu=0; iu<xaxis; iu++)
			    {
			      long global_index = (iu + (iv+isector*yaxis)*xaxis + iw*param.grid_size_x*param.grid_size_y)*sizeof(double);
			      long index = iu + iv*xaxis + iw*xaxis*yaxis;
			      fseek(file.pFilereal, global_index, SEEK_SET);
			      fwrite(&gridss_real[index], 1, sizeof(double), file.pFilereal);
			    }
		      for (int iw=0; iw<param.num_w_planes; iw++)
			for (int iv=0; iv<yaxis; iv++)
			  for (int iu=0; iu<xaxis; iu++)
			    {
			      long global_index = (iu + (iv+isector*yaxis)*xaxis + iw*param.grid_size_x*param.grid_size_y)*sizeof(double);
			      long index = iu + iv*xaxis + iw*xaxis*yaxis;
			      fseek(file.pFileimg, global_index, SEEK_SET);
			      fwrite(&gridss_img[index], 1, sizeof(double), file.pFileimg);
			    }
		    } 
		  else 
		    {
		      fwrite(gridss_real, size_of_grid/2, sizeof(double), file.pFilereal);
		      fwrite(gridss_img, size_of_grid/2, sizeof(double), file.pFileimg);
		    }
		  
		}
	     #else
	      /*
		for (int iw=0; iw<param.num_w_planes; iw++)
		for (int iv=0; iv<grid_size_y; iv++)
		for (int iu=0; iu<grid_size_x; iu++)
		{
		int isector = 0;
		long index = 2*(iu + iv*grid_size_x + iw*grid_size_x*grid_size_y);
		double v_norm = sqrt(gridtot[index]*gridtot[index]+gridtot[index+1]*gridtot[index+1]);
		fprintf (file.pFile, "%d %d %d %f %f %f\n", iu,iv,iw,gridtot[index],gridtot[index+1],v_norm);
		}
	      */
	     #endif
	      
	      fclose(file.pFilereal);
	      fclose(file.pFileimg);
	    }
	 #ifdef USE_MPI
	  MPI_Win_fence(0,writewin);
	  MPI_Win_free(&writewin);
	  MPI_Barrier(MPI_COMM_WORLD);
	 #endif

	  ADD_TIME(write, twt, tpr);
	 #endif //WRITE_DATA
	  
	  
	  // Phase correction  

	  

	  TAKE_TIME_START(phase);
          if(global_rank == 0) printf("PHASE CORRECTION\n");
	  double* image_real = (double*) calloc(xaxis*yaxis,sizeof(double));
	  double* image_imag = (double*) calloc(xaxis*yaxis,sizeof(double));
	  
	  phase_correction(gridss,image_real,image_imag,xaxis,yaxis,param.num_w_planes,param.grid_size_x,param.grid_size_y,resolution,metaData.wmin,metaData.wmax,param.num_threads);

	  TAKE_TIME_STOP(phase);
	  
	 #ifdef WRITE_IMAGE

	  TAKE_TIME(twt, tpr);
	  
	  if(global_rank == 0)
	    {
	      file.pFilereal = fopen (out.fftfile2,"wb");
	      file.pFileimg = fopen (out.fftfile3,"wb");
	      fclose(file.pFilereal);
	      fclose(file.pFileimg);
	    }
	 #ifdef USE_MPI
	  MPI_Barrier(MPI_COMM_WORLD);
	 #endif
	  if(global_rank == 0)printf("WRITING IMAGE\n");
	  for (int isector=0; isector<size; isector++)
	    {
	     #ifdef USE_MPI
	      MPI_Barrier(MPI_COMM_WORLD);
	     #endif
	      if(isector == global_rank)
		{
		  printf("%d writing\n",isector);
		  file.pFilereal = fopen (out.fftfile2,"ab");
		  file.pFileimg = fopen (out.fftfile3,"ab");
		  
		  long global_index = isector*(xaxis*yaxis)*sizeof(double);
	  
	  fseek(file.pFilereal, global_index, SEEK_SET);
	  fwrite(image_real, xaxis*yaxis, sizeof(double), file.pFilereal);
	  fseek(file.pFileimg, global_index, SEEK_SET);
	  fwrite(image_imag, xaxis*yaxis, sizeof(double), file.pFileimg);
	  
	  fclose(file.pFilereal);
	  fclose(file.pFileimg);
	}
    }
 #ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
 #endif

  ADD_TIME(write, twt, tpr);
 #endif //WRITE_IMAGE
  
 #endif  //FFTW
}
