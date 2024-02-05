#include <stdio.h>
#include <math.h>
#include "allvars.h"
#include "proto.h"


void gridding()
{
  
  if(global_rank == 0)printf("GRIDDING DATA\n");
  
  // Create histograms and linked lists

  TAKE_TIME_START(init);
  
  // Initialize array
  initialize_array();

  TAKE_TIME_STOP(init);
  TAKE_TIME_START(process);

  // Sector and Gridding data
  gridding_data();

  TAKE_TIME_STOP(process);
  
 #ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
 #endif

  return;
}

void initialize_array()
{

  histo_send     = (long*) calloc(nsectors+1,sizeof(long));
  int * boundary = (int*) calloc(metaData.Nmeasures,sizeof(int));
  double vvh;
    
  for (long iphi = 0; iphi < metaData.Nmeasures; iphi++)
    {
      boundary[iphi] = -1;
      vvh = data.vv[iphi];  //less or equal to 0.6
      int binphi = (int)(vvh*nsectors); //has values expect 0 and nsectors-1. So we use updist and downdist condition
      // check if the point influence also neighboring slabs
      double updist = (double)((binphi+1)*yaxis)*dx - vvh;
      double downdist = vvh - (double)(binphi*yaxis)*dx;
      //
      histo_send[binphi]++;
      if(updist < w_supporth && updist >= 0.0)
	{histo_send[binphi+1]++; boundary[iphi] = binphi+1;};
      if(downdist < w_supporth && binphi > 0 && downdist >= 0.0)
	{histo_send[binphi-1]++; boundary[iphi] = binphi-1;};
    }

  sectorarray = (long**)malloc ((nsectors+1) * sizeof(long*));
  for(int sec=0; sec<(nsectors+1); sec++)
    {
      sectorarray[sec] = (long*)malloc(histo_send[sec]*sizeof(long));
    }
    
  long *counter = (long*) calloc(nsectors+1,sizeof(long));
  for (long iphi = 0; iphi < metaData.Nmeasures; iphi++)
    {
      vvh = data.vv[iphi];
      int binphi = (int)(vvh*nsectors);
      double updist = (double)((binphi+1)*yaxis)*dx - vvh;
      double downdist = vvh - (double)(binphi*yaxis)*dx;
      sectorarray[binphi][counter[binphi]] = iphi;
      counter[binphi]++;
      if(updist < w_supporth && updist >= 0.0)
	{ sectorarray[binphi+1][counter[binphi+1]] = iphi; counter[binphi+1]++;};
      if(downdist < w_supporth && binphi > 0 && downdist >= 0.0)
	{ sectorarray[binphi-1][counter[binphi-1]] = iphi; counter[binphi-1]++;};
    }
     
  
 #ifdef VERBOSE
  for (int iii=0; iii<nsectors+1; iii++)printf("HISTO %d %d %ld\n", global_rank, iii, histo_send[iii]);
 #endif

  free( boundary);
  return;
}

void gridding_data()
{

  double shift = (double)(dx*yaxis);
  
 #ifdef USE_MPI
  MPI_Win_create(grid, size_of_grid*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &slabwin);
  MPI_Win_fence(0,slabwin);
 #endif

 #ifdef ONE_SIDE

   memset( (char*)Me.win.ptr, 0, size_of_grid*sizeof(double)*1.1);
   if( Me.Rank[myHOST] == 0 )
   {
       for( int tt = 1; tt < Me.Ntasks[myHOST]; tt++ )
           memset( (char*)Me.swins[tt].ptr, 0, size_of_grid*sizeof(double)*1.1);
   }

   MPI_Barrier(MPI_COMM_WORLD);

   if( Me.Rank[HOSTS] >= 0 )
       requests = (MPI_Request *)calloc( Me.Ntasks[WORLD], sizeof(MPI_Request) );

   #ifdef RING
   	if( Me.Rank[myHOST] == 0 ) {
       		*((int*)win_ctrl_hostmaster_ptr+CTRL_BARRIER_END) = 0;
       		*((int*)win_ctrl_hostmaster_ptr+CTRL_BARRIER_START) = 0; 
   	}

   	*((int*)Me.win_ctrl.ptr + CTRL_FINAL_STATUS) = FINAL_FREE;
   	*((int*)Me.win_ctrl.ptr + CTRL_FINAL_CONTRIB) = 0;
   	*((int*)Me.win_ctrl.ptr + CTRL_SHMEM_STATUS) = -1;
   	MPI_Barrier(*(Me.COMM[myHOST]));
   
   	blocks.Nblocks = Me.Ntasks[myHOST];
   	blocks.Bstart  = (int_t*)calloc( blocks.Nblocks, sizeof(int_t));
   	blocks.Bsize   = (int_t*)calloc( blocks.Nblocks, sizeof(int_t));
   	int_t size  = size_of_grid / blocks.Nblocks;
   	int_t rem   = size_of_grid % blocks.Nblocks;

   	blocks.Bsize[0]  = size + (rem > 0);
   	blocks.Bstart[0] = 0;
   	for(int b = 1; b < blocks.Nblocks; b++ ) {
   	blocks.Bstart[b] = blocks.Bstart[b-1]+blocks.Bsize[b-1];
   	blocks.Bsize[b] = size + (b < rem); }
   #endif

   #ifdef BINOMIAL
        copy_win_ptrs( (void***)&swins, Me.swins, Me.Ntasks[Me.SHMEMl] );
        copy_win_ptrs( (void***)&cwins, Me.scwins, Me.Ntasks[Me.SHMEMl] );

   	MPI_Barrier(MPI_COMM_WORLD);
       // printf("The no of task in shared memory %d, host %d\n", Me.Ntasks[Me.SHMEMl], Me.Ntasks[myHOST]);
        dsize_4 = (size_of_grid/4)*4;
        end_4   = (double*)Me.win.ptr + dsize_4;
        end_reduce  = (double*)Me.win.ptr + size_of_grid;

      	while( (1<< (++max_level) ) < Me.Ntasks[Me.SHMEMl] );
       // printf("Max level %d my rank %d\n",max_level, global_rank);      
      	*(int*)Me.win_ctrl.ptr     = DATA_FREE;
      	*((int*)Me.win_ctrl.ptr+1) = FINAL_FREE;
      	MPI_Barrier(*(Me.COMM[myHOST]));
   #endif
  
 #endif
  
 #ifndef USE_MPI
  file.pFile1 = fopen (out.outfile1,"w");
 #endif
  
  // calculate the resolution in radians
  resolution = 1.0/MAX(abs(metaData.uvmin),abs(metaData.uvmax));
  
  // calculate the resolution in arcsec 
  double resolution_asec = (3600.0*180.0)/MAX(abs(metaData.uvmin),abs(metaData.uvmax))/PI;
  if( global_rank == 0 )
    printf("RESOLUTION = %f rad, %f arcsec\n", resolution, resolution_asec);
  
  for (long isector = 0; isector < nsectors; isector++)
    {
      double twt, tpr;
      
      TAKE_TIME(twt, tpr);

      // define local destination sector
      //isector = (isector_count+rank)%size;  // this line must be wrong! [LT]
      
      // allocate sector arrays 
      long    Nsec       = histo_send[isector];
      double *uus        = (double*) malloc(Nsec*sizeof(double));
      double *vvs        = (double*) malloc(Nsec*sizeof(double));
      double *wws        = (double*) malloc(Nsec*sizeof(double));
      long    Nweightss  = Nsec*metaData.polarisations;
      long    Nvissec    = Nweightss*metaData.freq_per_chan;
      float *weightss    = (float*) malloc(Nweightss*sizeof(float));
      float *visreals    = (float*) malloc(Nvissec*sizeof(float));
      float *visimgs     = (float*) malloc(Nvissec*sizeof(float));
       
      // select data for this sector
      long icount = 0;
      long ip = 0;
      long inu = 0;

     #warning shall we omp-ize this ?
      for(long iphi = histo_send[isector]-1; iphi>=0; iphi--)
        {
	  long ilocal = sectorarray[isector][iphi];
	  uus[icount] = data.uu[ilocal];
	  vvs[icount] = data.vv[ilocal]-isector*shift;
	  wws[icount] = data.ww[ilocal];	  
	  UNROLL(4)
          PRAGMA_IVDEP
	  for (long ipol=0; ipol<metaData.polarisations; ipol++, ip++)
	    {
	      weightss[ip] = data.weights[ilocal*metaData.polarisations+ipol];
	    }
	  
	  PRAGMA_IVDEP
	  UNROLL(4)
	    for (long ifreq=0; ifreq<metaData.polarisations*metaData.freq_per_chan; ifreq++, inu++)
	      {
		visreals[inu] = data.visreal[ilocal*metaData.polarisations*metaData.freq_per_chan+ifreq];
		visimgs[inu] = data.visimg[ilocal*metaData.polarisations*metaData.freq_per_chan+ifreq];
	      //if(visimgs[inu]>1e10 || visimgs[inu]<-1e10)printf("%f %f %ld %ld %d %ld %ld\n",visreals[inu],visimgs[inu],inu,Nvissec,rank,ilocal*metaData.polarisations*metaData.freq_per_chan+ifreq,metaData.Nvis);
	    }
	  icount++;
	}

      ADD_TIME(compose, twt, tpr);

     #ifndef USE_MPI
      double vvmin = 1e20;
      double uumax = -1e20;
      double vvmax = -1e20;

     #warning shall we omp-ize this ?
      for (long ipart=0; ipart<Nsec; ipart++)
	{
	  uumin = MIN(uumin,uus[ipart]);
	  uumax = MAX(uumax,uus[ipart]);
	  vvmin = MIN(vvmin,vvs[ipart]);
	  vvmax = MAX(vvmax,vvs[ipart]);
	     
	  if(ipart%10 == 0)fprintf (file.pFile, "%ld %f %f %f\n",isector,uus[ipart],vvs[ipart]+isector*shift,wws[ipart]);
	}
	 
      printf("UU, VV, min, max = %f %f %f %f\n", uumin, uumax, vvmin, vvmax);
     #endif

      // Make convolution on the grid

     #ifdef VERBOSE
      printf("Processing sector %ld\n",isector);
     #endif
      TAKE_TIME(twt, tpr);

      wstack(param.num_w_planes,
	     Nsec,
	     metaData.freq_per_chan,
	     metaData.polarisations,
	     uus,
	     vvs,
	     wws,
	     visreals,
	     visimgs,
	     weightss,
	     dx,
	     dw,
	     param.w_support,
	     xaxis,
	     yaxis,
	     gridss,
	     param.num_threads);

      ADD_TIME(kernel, twt, tpr);
      
      /* int z =0 ;
       * #pragma omp target map(to:test_i_gpu) map(from:z)
       * {
       *   int x; // only accessible from accelerator
       *     x = 2;
       *       z = x + test_i_gpu;
       *       }*/

     #ifdef VERBOSE
      printf("Processed sector %ld\n",isector);
     #endif
      /* ----------------
       * REDUCE
       * ---------------- */

      double twt_r, tpr_r;
      TAKE_TIME(twt_r, tpr_r);
                                                     // ..................
     #ifndef USE_MPI                                 // REDUCE WITH NO MPI                
      
      #pragma omp parallel
      {
	long stride = isector * size_of_grid;
       #pragma omp for
	for (long iii=0; iii< size_fo_grid; iii++)
	  gridtot[stride+iii] = gridss[iii];
      }

                                                     // ..................
                                                     // REDUCE WITH MPI
     #else

      // Write grid in the corresponding remote slab      
      // int target_rank = (int)isector;    it implied that size >= nsectors
      int target_rank = (int)(isector % size);
       
     #ifdef ONE_SIDE

     printf("One Side communication active\n");

     #ifdef RING
     double _twt_;
     TAKE_TIMEwt(_twt_);
     int res = reduce_ring(target_rank);
     ADD_TIMEwt(reduce_ring, _twt_);
     #endif

     #ifdef BINOMIAL
     int res = reduce_binomial(target_rank);
     #endif


     #else   // relates to #ifdef ONE_SIDE
      
      {
	double _twt_;
	TAKE_TIMEwt(_twt_);
	MPI_Reduce(gridss,grid,size_of_grid,MPI_DOUBLE,MPI_SUM,target_rank,MPI_COMM_WORLD);
	ADD_TIMEwt(mpi, _twt_);
      }
      
     #endif  //  closes #ifdef ONE_SIDE
     #endif  //  closes USE_MPI

      ADD_TIME(reduce, twt_r, tpr_r);

      
      // Deallocate all sector arrays
      free(uus);
      free(vvs);
      free(wws);
      free(weightss);
      free(visreals);
      free(visimgs);
      // End of loop over sector    
    }
    #ifdef ONE_SIDE
    #ifdef RING

         for( int jj = 0; jj < size_of_grid; jj++)
         {
       	    	*((double*)grid+jj) = *((double*)Me.fwin.ptr+jj);
         }

    #endif
    #endif

  free( histo_send );

 #ifndef USE_MPI
  fclose(file.pFile1);
 #endif

 #ifdef USE_MPI
 // MPI_Win_fence(0,slabwin);
 #ifdef ONE_SIDE
  numa_shutdown(global_rank, 0, &MYMPI_COMM_WORLD, &Me);
 #endif
  MPI_Barrier(MPI_COMM_WORLD);
 #endif
  
}

void write_gridded_data()
{

   #ifdef WRITE_DATA
     // Write results
     if (global_rank == 0)
     {
        printf("WRITING GRIDDED DATA\n");
        file.pFilereal = fopen (out.outfile2,"wb");
        file.pFileimg = fopen (out.outfile3,"wb");
	
       #ifdef USE_MPI
	for (int isector=0; isector<nsectors; isector++)
	  {
	    MPI_Win_lock(MPI_LOCK_SHARED,isector,0,slabwin);
	    MPI_Get(gridss,size_of_grid,MPI_DOUBLE,isector,0,size_of_grid,MPI_DOUBLE,slabwin);
	    MPI_Win_unlock(isector,slabwin);
	    for (long i=0; i<size_of_grid/2; i++)
              {
		gridss_real[i] = gridss[2*i];
		gridss_img[i] = gridss[2*i+1];
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
			//double v_norm = sqrt(gridss[index]*gridss[index]+gridss[index+1]*gridss[index+1]);
			//fprintf (file.pFile, "%d %d %d %f %f %f\n", iu,isector*yaxis+iv,iw,gridss[index],gridss[index+1],v_norm);
		      }
		
              }
	    else
              {
		for (int iw=0; iw<param.num_w_planes; iw++)
		  {
		    long global_index = (xaxis*isector*yaxis + iw*param.grid_size_x*param.grid_size_y)*sizeof(double);
		    long index = iw*xaxis*yaxis;
		    fseek(file.pFilereal, global_index, SEEK_SET);
		    fwrite(&gridss_real[index], xaxis*yaxis, sizeof(double), file.pFilereal);
		    fseek(file.pFileimg, global_index, SEEK_SET);
		    fwrite(&gridss_img[index], xaxis*yaxis, sizeof(double), file.pFileimg);
		  }
              }
          }
       #else
	for (int iw=0; iw<param.num_w_planes; iw++)
	  for (int iv=0; iv<param.grid_size_y; iv++)
	    for (int iu=0; iu<param.grid_size_x; iu++)
	      {
		long index = 2*(iu + iv*param.grid_size_x + iw*param.grid_size_x*param.grid_size_y);
		fwrite(&gridtot[index], 1, sizeof(double), file.pFilereal);
		fwrite(&gridtot[index+1], 1, sizeof(double), file.pFileimg);
		//double v_norm = sqrt(gridtot[index]*gridtot[index]+gridtot[index+1]*gridtot[index+1]);
		//fprintf (file.pFile, "%d %d %d %f %f %f\n", iu,iv,iw,gridtot[index],gridtot[index+1],v_norm);
	      }
       #endif
        fclose(file.pFilereal);
        fclose(file.pFileimg);
     }
     
    #ifdef USE_MPI
     MPI_Win_fence(0,slabwin);
    #endif
     
    #endif //WRITE_DATA      
}

void copy_win_ptrs( void ***A, win_t *B, int n)
{
  if ( *A == NULL )
    *A = (void**)malloc( n*sizeof(void*));
  for( int i = 0; i < n; i++ )
    (*A)[i] = (void*)B[i].ptr;
  return;
}

