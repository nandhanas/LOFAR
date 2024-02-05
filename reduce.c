#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif

//Revamped by NANDHANA SAKTHIVEL as part of her Master thesis, DSSC, UNITS, Italy


#include <stdio.h>
#include <math.h>
#include "allvars.h"
#include "proto.h"
//#include "mypapi.h"


#if defined(DEBUG)
double check_host_value ;
double check_global_value ;
#endif

//struct { double rtime, ttotal, treduce, tspin, tspin_in, tmovmemory, tsum;} timing = {0};
//struct { double tmpi, tmpi_reduce, tmpi_reduce_wait, tmpi_setup;} timingmpi = {0};


int_t summations = 0;
int_t memmoves   = 0;

int reduce_ring (int target_rank)
{
	/* -------------------------------------------------
	 *
	 *  USE THE SHARED MEMORY WINDOWS TO REDUCE DATA 
	 * ------------------------------------------------- */
	
	{
	  timing.rtime  = CPU_TIME_rt;
	  timing.ttotal = CPU_TIME_pr;

          #ifdef DEBUG
                check_host_value = 0;
		for( int jj = 0; jj < Me.Ntasks[myHOST]; jj++ )
		{
			check_host_value += (double)(Me.Ranks_to_myhost[jj]);				
		}
          #endif
         
	 #pragma omp parallel num_threads(2)
	  {
	    int thid         = omp_get_thread_num();
	    int Ntasks_local = Me.Ntasks[Me.SHMEMl];

		if( thid == 1 )
		  {		    
		                                                                       // check that the data in Me.win
		                                                                       // can be overwritten by new data
		                                                                       // -> this condition is true when
		                                                                       // win_ctrl has the value "DATA_FREE"
		    		    
		    if( Ntasks_local > 1 )
		      {
                       	for( int jj = 0; jj < size_of_grid; jj++ )
                          *((double*)Me.win.ptr+jj) = *((double*)gridss+jj);

			int value = target_rank * Ntasks_local;
			
			for ( int jj = 0; jj < Me.Ntasks[Me.SHMEMl]; jj++ )
			  *((int*)Me.win_ctrl.ptr+CTRL_BLOCKS+jj) = value;

			atomic_store((int*)Me.win_ctrl.ptr+CTRL_FINAL_CONTRIB, 0);
			//atomic_thread_fence(memory_order_release);
			atomic_store((int*)Me.win_ctrl.ptr+CTRL_SHMEM_STATUS, value);
			//CPU_TIME_STAMP( Me.Rank[myHOST], "A0");
						                                       // calls the reduce
			double start = CPU_TIME_tr;			
			int ret = shmem_reduce_ring( target_rank, target_rank, size_of_grid, &Me, (double*)Me.win.ptr, &blocks );	
			timing.treduce += CPU_TIME_tr - start;
		         	
			if( ret != 0 )
			  {
			    printf("Task %d : shared-memory reduce for sector %d has returned "
				   "an error code %d : better stop here\n",
				   global_rank, target_rank, ret );
			    free( blocks.Bsize );
			    free( blocks.Bstart );
			    numa_shutdown(global_rank, 0, &MYMPI_COMM_WORLD, &Me);
			    MPI_Finalize();
			  }
			
		      }
		    else
		      {
			ACQUIRE_CTRL((int*)Me.win_ctrl.ptr+CTRL_FINAL_STATUS, FINAL_FREE, timing.tspin, != );
                                                                 		       // mimic the production of new data
                       for( int jj = 0; jj < size_of_grid; jj++ )
		       {
                          *((double*)Me.fwin.ptr+jj) = *((double*)gridss+jj);
			  *((double*)Me.win.ptr+size_of_grid+jj) = *((double*)gridss+jj);
		       }
			atomic_store(((int*)Me.win_ctrl.ptr+CTRL_FINAL_CONTRIB), Ntasks_local);
		      }

		    int Im_target                   = (global_rank == target_rank);
		    int Im_NOT_target_but_Im_master = (Me.Nhosts>1) &&
		      (Me.Ranks_to_host[target_rank]!=Me.myhost) && (Me.Rank[myHOST]==0);
		    
		    if( Im_target || Im_NOT_target_but_Im_master )
		      {			
			ACQUIRE_CTRL((int*)Me.win_ctrl.ptr+CTRL_FINAL_CONTRIB, Ntasks_local, timing.tspin, !=);
                        
			atomic_store(((int*)Me.win_ctrl.ptr+CTRL_FINAL_STATUS), target_rank);
		      }

		    atomic_fetch_add( (int*)win_ctrl_hostmaster_ptr+CTRL_BARRIER_END, (int)1 );
		    switch( Me.Rank[Me.SHMEMl] ) {
		    case 0: { ACQUIRE_CTRL((int*)win_ctrl_hostmaster_ptr+CTRL_BARRIER_END, Ntasks_local, timing.tspin, !=);
			atomic_store( (int*)win_ctrl_hostmaster_ptr+CTRL_BARRIER_END, (int)0 ); } break;
		    default : ACQUIRE_CTRL((int*)win_ctrl_hostmaster_ptr+CTRL_BARRIER_END, 0, timing.tspin, !=); break;
		    }

		  }

		else 
		  {
		    
		    /* 
		     *
		     *  REDUCE AMONG HOSTS
		     */

		    if ( (Me.Nhosts > 1) && (Me.Rank[myHOST] == 0) )
		      {			
			double start = CPU_TIME_tr;
			
			int target_task       = Me.Ranks_to_host[target_rank];
			int Im_hosting_target = Me.Ranks_to_host[target_rank] == Me.myhost;
			int target            = 0;
			
			if( Im_hosting_target )
			  while( (target < Me.Ntasks[Me.SHMEMl]) &&
				 (Me.Ranks_to_myhost[target] != target_rank) )
			    target++;

			
			int    *ctrl_ptr    = (int*)Me.scwins[target].ptr+CTRL_FINAL_STATUS;

			double *send_buffer = ( Im_hosting_target ? MPI_IN_PLACE : (double*)Me.win.ptr+size_of_grid );
			double *recv_buffer = ( Im_hosting_target ? (double*)Me.sfwins[target].ptr : NULL );


			timingmpi.tmpi_setup += CPU_TIME_tr - start;

			double tstart = CPU_TIME_tr;
			
			ACQUIRE_CTRL( ctrl_ptr, target_rank, timing.tspin, !=);
			
			timingmpi.tmpi_reduce_wait += CPU_TIME_tr - tstart;

			tstart = CPU_TIME_tr;
			MPI_Ireduce(send_buffer, recv_buffer, size_of_grid, MPI_DOUBLE, MPI_SUM, target_task, COMM[HOSTS], &requests[target_rank]);			
			timingmpi.tmpi_reduce += CPU_TIME_tr - tstart;
			MPI_Wait( &requests[target_rank], MPI_STATUS_IGNORE );
			atomic_store(ctrl_ptr, FINAL_FREE);
			timingmpi.tmpi += CPU_TIME_tr - start;
		      }

		    atomic_thread_fence(memory_order_release);
		    
		  } // closes thread 0
		
        
	 }
	  timing.rtime  = CPU_TIME_rt - timing.rtime;
	  timing.ttotal = CPU_TIME_pr - timing.ttotal;
	  
	}

  return 0;
}

int reduce_binomial ( int target_rank )
{


	/* -------------------------------------------------
  	 *
  	 *  USE THE SHARED MEMORY WINDOWS TO REDUCE DATA
 	 * ------------------------------------------------- */

      {
	  timing.rtime  = CPU_TIME_rt;
	  timing.ttotal = CPU_TIME_pr;
	 #pragma omp parallel num_threads(2)
	  {
	    int thid = omp_get_thread_num();



		if( thid == 1 )
		  {
		                                                                       // check that the data in Me.win
		                                                                       // can be overwritten by new data 		                                                                       		                                                                       // -> this condition is true when                                                          		                                                                       		                                                                       // win_ctrl has the value "DATA_FREE"

                      ACQUIRE_CTRL((int*)Me.win_ctrl.ptr, DATA_FREE, timing.tspin, != )
                      memcpy(Me.win.ptr, gridss, sizeof(gridss));
                      if( Me.Ntasks[myHOST] > 1 )
		      {
				int value = target_rank * (max_level+1);
				atomic_store((int*)Me.win_ctrl.ptr, value);
			
				double start = CPU_TIME_tr;
                               // printf("Im before shmem_reduce my rank %d target rank %d size_of_grid %d\n", global_rank, target_rank, size_of_grid);
				int ret = shmem_reduce_binomial( target_rank, target_rank, size_of_grid, &Me, (double*)Me.win.ptr, max_level );
				//printf("Im after shmem_reduce my rank %d target rank %d\n", global_rank, target_rank);
				timing.treduce += CPU_TIME_tr - start;
				if( ret != 0 )
			  	{
			    		printf("Task %d : shared-memory reduce for sector %d has returned "
				   		"an error code %d : better stop here\n",
				  		 global_rank, target_rank, ret );
			    		free(cwins);
			    		free(swins);
			    		numa_shutdown(global_rank, 0, &MYMPI_COMM_WORLD, &Me);
			    		MPI_Finalize();
			  	}
				
		      }
		    else
		      atomic_store((int*)Me.win_ctrl.ptr, DATA_FREE);

		    int Im_target = (global_rank == target_rank);
		    int Im_NOT_target_but_Im_master = (Me.Nhosts>1) &&
		      (Me.Ranks_to_host[target_rank]!=Me.myhost) && (Me.Rank[myHOST]==0);

                    if( Im_target || Im_NOT_target_but_Im_master )
		      {
			ACQUIRE_CTRL((int*)Me.win_ctrl.ptr+1, FINAL_FREE, timing.tspin, != );
			double start = CPU_TIME_tr;
			double * restrict final = (double*)Me.win.ptr + size_of_grid;
			double * restrict run   = (double*)Me.win.ptr;
			for( ; run < end_4; run += 4, final += 4 ) {
			  *final     = *run;
			  *(final+1) = *(run+1);
			  *(final+2) = *(run+2);
			  *(final+3) = *(run+3); }
			for( ; run < end_reduce; run++, final++ )
			  *final = *run;
			timing.tmovmemory += CPU_TIME_tr - start;
                         printf("Im inside I'm target my rank %d target rank %d\n", global_rank, target_rank);

			atomic_store(((int*)Me.win_ctrl.ptr+1), target_rank);
			atomic_store((int*)Me.win_ctrl.ptr, DATA_FREE);
			atomic_thread_fence(memory_order_release);
		      }

		  }
                  else
		  {
		    //MPI_Barrier(*Me.COMM[myHOST]);
		    /*
		     *
		     *  REDUCE AMONG HOSTS
		     */

                     if ( (Me.Nhosts > 1) && (Me.Rank[myHOST] == 0) )
		      {
			double start = CPU_TIME_tr;

			int target_task       = Me.Ranks_to_host[target_rank];
			int Im_hosting_target = Me.Ranks_to_host[target_rank] == Me.myhost;
			int target            = 0;

			if( Im_hosting_target )
			  while( (target < Me.Ntasks[Me.SHMEMl]) &&
				 (Me.Ranks_to_myhost[target] != target_rank) )
			    target++;


			int    *ctrl_ptr    = ( target == 0 ? (int*)Me.win_ctrl.ptr+1 : ((int*)Me.scwins[target].ptr)+1 );

			double *send_buffer = ( Im_hosting_target ? (double*)Me.swins[target].ptr+size_of_grid :
						(double*)Me.win.ptr+size_of_grid );
			double *recv_buffer = ( Im_hosting_target ? (double*)Me.sfwins[target].ptr : NULL );

			timingmpi.tmpi_setup += CPU_TIME_tr - start;

			double tstart = CPU_TIME_tr;

			ACQUIRE_CTRL( ctrl_ptr, target_rank, timing.tspin, != );

			timingmpi.tmpi_reduce_wait += CPU_TIME_tr - tstart;

			tstart = CPU_TIME_tr;
			MPI_Ireduce(send_buffer, recv_buffer, size_of_grid, MPI_DOUBLE, MPI_SUM, target_task, COMM[HOSTS], &requests[target_rank]);
			timingmpi.tmpi_reduce += CPU_TIME_tr - tstart;

			MPI_Wait( &requests[target_rank], MPI_STATUS_IGNORE );
			atomic_store(ctrl_ptr, FINAL_FREE);

			iter++;
                        timingmpi.tmpi += CPU_TIME_tr - start;
			fflush(stdout);
		      }

		  } // closes thread 0
                  atomic_thread_fence(memory_order_release);


	  }
	  timing.rtime  = CPU_TIME_rt - timing.rtime;
	  timing.ttotal = CPU_TIME_pr - timing.ttotal;

	  free(cwins);
	  free(swins);


	}

  return 0;
}
      

int shmem_reduce_ring( int sector, int target_rank, int_t size_of_grid, map_t *Me, double * restrict data, blocks_t *blocks )
 {


   int local_rank            = Me->Rank[Me->SHMEMl];
   int target_rank_on_myhost = 0;
   int Im_hosting_target     = 0;
   
   if( Me->Ranks_to_host[ target_rank ] == Me->myhost )
     // exchange rank 0 with target rank
     // in this way the following log2 alogorithm,
     // which reduces to rank 0, will work for
     // every target rank
     {

       Im_hosting_target = 1;
       target_rank_on_myhost = 0;
       while( (target_rank_on_myhost < Me->Ntasks[Me->SHMEMl]) &&
	      (Me->Ranks_to_myhost[target_rank_on_myhost] != target_rank) )
	 target_rank_on_myhost++;

       if( target_rank_on_myhost == Me->Ntasks[Me->SHMEMl] )
	 return -1;
     }

   // Here we start the reduction
   //

   dprintf(1, 0, 0, "@ SEC %d t %d (%d), %d\n",
	   sector, local_rank, global_rank, *(int*)Me->win_ctrl.ptr);

   // main reduction loop
   //
   int SHMEMl  = Me->SHMEMl;
   int Nt      = Me->Ntasks[SHMEMl];
   int end     = Me->Ntasks[SHMEMl]-1;
   int target  = (Nt+(local_rank-1)) % Nt;
   int myblock = local_rank;
   int ctrl    = sector*Nt;

   //CPU_TIME_STAMP( local_rank, "R0");
   ACQUIRE_CTRL( ((int*)Me->scwins[target].ptr)+CTRL_SHMEM_STATUS, ctrl, timing.tspin_in, != );        // is my target ready?
   
   for(int t = 0; t < end; t++)
     {
 	                                                                      // prepare pointers for the summation loop
       int_t  dsize = blocks->Bsize[myblock];
       double * restrict my_source = (double*)Me->swins[target].ptr + blocks->Bstart[myblock];
       double * restrict my_target = data + blocks->Bstart[myblock];
       my_source = __builtin_assume_aligned( my_source, 8);
       my_target = __builtin_assume_aligned( my_target, 8);

     //  dprintf(1, 0, 0, "+ SEC %d host %d l %d t %d <-> %d block %d from %llu to %llu\n",
//	       sector, Me->myhost, t, local_rank, target, myblock, 
//	       blocks->Bstart[myblock], blocks->Bstart[myblock]+dsize );
       
	                                                                      // check whether the data of the source rank
	                                                                      // are ready to be used (control tag must have
	                                                                      // the value of the current sector )
       //CPU_TIME_STAMP( local_rank, "R1");
       ACQUIRE_CTRL( ((int*)Me->scwins[target].ptr)+CTRL_BLOCKS+myblock, ctrl, timing.tspin_in, !=);        // is myblock@Me ready?
       //CPU_TIME_STAMP( local_rank, "R2");

	                                                                      // performs the summation loop
	                                                                      //
      #if defined(USE_PAPI)
       if( sector == 0 ) {
	 PAPI_START_CNTR;
	 summations += dsize; }
      #else
       summations += dsize;
      #endif

       double  tstart = CPU_TIME_tr;
       double *my_end = my_source+dsize;
       switch( dsize < BUNCH_FOR_VECTORIZATION )
	 {
	 case 0: {
	   int      dsize_4  = (dsize/4)*4;
	   double * my_end_4 = my_source+dsize_4;
	   for( ; my_source < my_end_4; my_source+=4, my_target+=4 )
	     {
	       __builtin_prefetch( my_target+8, 0, 1);
	       __builtin_prefetch( my_source+8, 0, 1);
	       *my_target += *my_source;
	       *(my_target+1) += *(my_source+1);
	       *(my_target+2) += *(my_source+2);
	       *(my_target+3) += *(my_source+3);
//                printf("The rank %d target value of 3 %lf\n", global_rank, (target+3));
	     } }
	 case 1: { for( ; my_source < my_end; my_source++, my_target++)
	       *my_target += *my_source; } break;
	 }
       
       timing.tsum += CPU_TIME_tr - tstart;
      #if defined(USE_PAPI)
       if( sector == 0 )
	 PAPI_STOP_CNTR;
      #endif

       
       ctrl++;
       atomic_store( ((int*)Me->win_ctrl.ptr+CTRL_BLOCKS+myblock), ctrl );
       //CPU_TIME_STAMP( local_rank, "R3");
  //     dprintf(1, 0, 0, "- SEC %d host %d l %d t %d ... writing tag %d on block %d = %d\n",
//	       sector, Me->myhost, t, local_rank, ctrl, myblock, 
//	       *((int*)Me->win_ctrl.ptr+CTRL_BLOCKS+myblock) );
       
       myblock = (Nt+(myblock-1)) % Nt;
       atomic_thread_fence(memory_order_release);
     }

   myblock = (myblock+1)%Nt;
   int_t offset = blocks->Bstart[myblock];
   int_t dsize  = blocks->Bsize[myblock];

 //  dprintf(1,0,0, "c SEC %d host %d t %d (%d) ==> t %d, block %d %llu from %llu\n",
//	   sector, Me->myhost, local_rank, global_rank, target_rank_on_myhost, myblock, dsize, offset );

   double tstart2 = CPU_TIME_tr;
   double * restrict my_source = data+offset;
   double *          my_end    = my_source+dsize;
   double * restrict my_final;

   switch( Im_hosting_target ) {
   case 0: my_final = (double*)Me->swins[0].ptr+size_of_grid+offset; break;
   case 1: my_final = (double*)Me->sfwins[target_rank_on_myhost].ptr+offset; }

   my_source = __builtin_assume_aligned( my_source, 8);
   my_final  = __builtin_assume_aligned( my_final, 8);

   atomic_thread_fence(memory_order_acquire);
   ACQUIRE_CTRL((int*)Me->scwins[target_rank_on_myhost].ptr+CTRL_FINAL_STATUS, FINAL_FREE, timing.tspin_in, != );

   switch( dsize < BUNCH_FOR_VECTORIZATION ) {
   case 0: { double *end_4 = my_source + (dsize/4)*4;
       for( ; my_source < end_4; my_source+=4, my_final+=4) {
	 *my_final = *my_source; *(my_final+1) = *(my_source+1);
	 *(my_final+2) = *(my_source+2); *(my_final+3) = *(my_source+3); } }
   case 1: { for ( ; my_source < my_end; my_source++, my_final++ ) *my_final = *my_source; } break;
   }
   
   atomic_fetch_add((int*)Me->scwins[target_rank_on_myhost].ptr+CTRL_FINAL_CONTRIB, (int)1);   
   timing.tmovmemory += CPU_TIME_tr - tstart2;

   memmoves += dsize;
   
   //atomic_thread_fence(memory_order_release);
   
   return 0;
 }

int shmem_reduce_binomial( int sector, int target_rank, int dsize, map_t *Me, double * restrict data, int max_level )
 {

   //printf("Im inside shmem_reduce_binomial my rank %d target rank %d\n", global_rank, target_rank);
   int local_rank = Me->Rank[Me->SHMEMl];
   int target_rank_on_myhost = -1;

   if( Me->Ranks_to_host[ target_rank ] == Me->myhost )
     // exchange rank 0 with target rank
     // in this way the following log2 alogorithm,
     // which reduces to rank 0, will work for
     // every target rank
     {

       target_rank_on_myhost = 0;
       while( (target_rank_on_myhost < Me->Ntasks[Me->SHMEMl]) &&
	      (Me->Ranks_to_myhost[target_rank_on_myhost] != target_rank) )
	 target_rank_on_myhost++;

       if( target_rank_on_myhost > 0 )
       // the target is not the task that already has rank 0
       // on my host
       {
	   if( local_rank == 0 )
	     local_rank = target_rank_on_myhost;
	   else if( local_rank == target_rank_on_myhost )
	     local_rank = 0;

	   void *temp = (void*)swins[target_rank_on_myhost];
	   swins[target_rank_on_myhost] = swins[0];
	   swins[0] = (double*)temp;

	   temp = (void*)cwins[target_rank_on_myhost];
	   cwins[target_rank_on_myhost] = cwins[0];
	   cwins[0] = (int*)temp;
	 }

       if( target_rank_on_myhost == Me->Ntasks[Me->SHMEMl] )
	 return -1;
      }
     // printf("Im after ist if  shmem_reduce_binomial my rank %d target rank %d\n", global_rank, target_rank);
      // Here we start the reduction
      //    

      int dsize2   = dsize / 2;
      int dsize2_4 = (dsize2/4)*4;

      int my_maxlevel = max_level;
      while( (local_rank % (1<<my_maxlevel)) ) my_maxlevel--;

      //printf("my max_level %d max level %d my rank %d\n", my_maxlevel, max_level, global_rank);
      //dprintf(1, 0, 0, "@ SEC %d t %d (%d), %d %d\n",sector, local_rank, global_rank, *(int*)Me->win_ctrl.ptr, my_maxlevel);
     
      // main reduction loop
      //    
      for(int l = 0; l <= my_maxlevel; l++)
      {
       int threshold = 1 << l;
       int source    = local_rank ^ (1<<l);
    
       if( ( local_rank % threshold == 0 ) &&
	   ( source < Me->Ntasks[Me->SHMEMl] ) )

	 // a task enters here only if it is a
	 // multiple of threshold AND if its source
	 // is not beyond the tasks array
	 // 
	 {
	   int I_m_target = local_rank < source;
	//   printf("Im inside the 1st if of reduction for loop inside shmem_reduce_binomial my rank %d target rank %d I'm target %d source %d\n", global_rank, target_rank, I_m_target, source);
 	                                                                       // prepare pointers for the summation loop
 	   double * restrict my_source = ( I_m_target ? swins[source] : data + dsize2);
	   double * restrict my_target = ( I_m_target ? data : swins[source]+dsize2 );
	   my_source = __builtin_assume_aligned( my_source, 8);
	   my_target = __builtin_assume_aligned( my_target, 8);
        //   printf("Im inside the 1st if of reduction after source and target assignment for loop inside shmem_reduce_binomial my rank %d target rank %d\n", global_rank, target_rank);
	  #if defined(DEBUG)
	   int my_start = ( I_m_target ? 0 : dsize2);	   
	  #endif

	                                                                      // check whether the data of the source rank
	                                                                      // are ready to be used (control tag must have                  	                                                                      	                                                                 // the value of the current sector )
	  int ctrl = sector*(max_level+1)+l;
	   ACQUIRE_CTRL( cwins[source], ctrl, timing.tspin_in, < );
          //  printf("Im inside the 1st if of reduction after aquire ctrl for loop inside shmem_reduce_binomial my rank %d target rank %d\n", global_rank, target_rank);
	                                                                      // performs the summation loop
	                                                                      // 	                                                                      
	   double * my_end   = my_source+dsize2;
	   dprintf(1, 0, 0, "+ SEC %d l %d t %d <-> %d from %d to %ld\n",
		   sector, l, local_rank, source, my_start, my_end - my_source + my_start);

	   
	  #if defined(USE_PAPI)
	   if( sector == 0 )
	     PAPI_START_CNTR;
	  #endif
	   double tstart = CPU_TIME_tr;
	   summations += (!sector)*dsize2;
	   if( dsize2 < 16 )
	     {
            //   printf("Im inside the  if dsize2<16 of reduction after aquire ctrl for loop inside shmem_reduce_binomial my rank %d target rank %d\n", global_rank, target_rank);
	       for( ; my_source < my_end; my_source++, my_target++)
		 *my_target += *my_source;
             //  printf("Im inside the  if dsize2<16 of reduction after  for loop inside shmem_reduce_binomial my rank %d target rank %d\n", global_rank, target_rank);
	     }
	   else
	     {
             //  printf("Im inside the  else dsize2<16 of reduction after aquire ctrl for loop inside shmem_reduce_binomial my rank %d target rank %d\n", global_rank, target_rank);	       
	       double * my_end_4 = my_source+dsize2_4;
	     //   printf("Im inside the  else dsize2<16 of reduction after my_end_4 for loop inside shmem_reduce_binomial my rank %d target rank %d\n", global_rank, target_rank);    
	       for( ; my_source < my_end_4; my_source+=4, my_target+=4 )
		 {
               //     printf("I'm inside the beginning of the for loop for adding source my source %lf rank %d target rank %d\n",*my_source, global_rank, target_rank);
		    *my_target += *my_source;
		   *(my_target+1) += *(my_source+1);
		   *(my_target+2) += *(my_source+2);
	           *(my_target+3) += *(my_source+3);
                //   printf("I'm inside the for loop for adding source my source %lf rank %d target rank %d\n",*my_source, global_rank, target_rank);
		 }
             //  printf("Im inside the  else dsize2<16 of reduction after 1st for loop inside shmem_reduce_binomial my rank %d target rank %d\n", global_rank, target_rank);
	       for( ; my_source < my_end; my_source++, my_target++)
		 *my_target += *my_source;	       
              //  printf("Im inside the  else dsize2<16 of reduction after 2st for loop inside shmem_reduce_binomial my rank %d target rank %d\n", global_rank, target_rank);
	     }
            //printf("Im inside the 1st if of reduction after summation for loop inside shmem_reduce_binomial my rank %d target rank %d\n", global_rank, target_rank);
	   timing.tsum += CPU_TIME_tr - tstart;
	  #if defined(USE_PAPI)
	   if( sector == 0 )
	     PAPI_STOP_CNTR;
	  #endif

	   atomic_thread_fence(memory_order_release);
	   

	   int value = (sector*(max_level+1)+l+1);
	   switch( I_m_target ) {
	   case 0: atomic_store((int*)Me->win_ctrl.ptr, value); break;
	   case 1: { ACQUIRE_CTRL( cwins[source], value, timing.tspin_in, != );
	       atomic_store(cwins[source], DATA_FREE);
	       atomic_store((int*)(Me->win_ctrl.ptr), value); } break;
	   }

	   dprintf(1,0,0, "- SEC %d l %d t %d <-> %d done : %d\n",
		   sector, l, local_rank, source, *(int*)(Me->win_ctrl.ptr));
	   //printf("Im at the end of reduction for loop inside shmem_reduce_binomial my rank %d target rank %d\n", global_rank, target_rank);
         }
       
       else
	 // .. otherwise, the task has ended its participation
	 // into the reduction loop
	 {
	   atomic_store((int*)(Me->win_ctrl.ptr), my_maxlevel);
	 }
	   
         atomic_thread_fence(memory_order_release);
      }
     // printf("Im after reduction for loop inside shmem_reduce_binomial my rank %d target rank %d\n", global_rank, target_rank);

      if ( target_rank_on_myhost > 0 )
      {
         void *temp = (void*)swins[target_rank_on_myhost];
         swins[target_rank_on_myhost] = swins[0];
         swins[0] = (double*)temp;

         temp = (void*)cwins[target_rank_on_myhost];
         cwins[target_rank_on_myhost] = cwins[0];
         cwins[0] = (int*)temp;
       //  printf("Im inside targetrankonmyhost %d  inside shmem_reduce_binomial my rank %d target rank %d\n", target_rank_on_myhost ,global_rank, target_rank);
      }

   return 0;
 }


