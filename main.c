#include<stdio.h>
#include "allvars.h"
#include "proto.h"

//Revamped by NANDHANA SAKTHIVEL as part of her Master thesis, DSSC, UNITS, Italy

// Main Code
int main(int argc, char * argv[])
{


        if(argc > 1)
        {
          strcpy(in.paramfile,argv[1]);
        }
        else
        {
          fprintf(stderr, "Parameter file is not given\n");
          exit(1);
        }
 
        /* Initializing MPI Environment */

        #ifdef USE_MPI
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if(global_rank == 0)printf("Running with %d MPI tasks\n",size);
       #ifdef USE_FFTW
	fftw_mpi_init();
       #endif
	MPI_Comm_dup(MPI_COMM_WORLD, &MYMPI_COMM_WORLD);
       #else
		global_rank = 0;
		size = 1;
	#endif
	#ifdef ACCOMP
	  if(global_rank == 0){
  		if (0 == omp_get_num_devices()) {
      			printf("No accelerator found ... exit\n");
      			exit(255);
  		 }
   		printf("Number of available GPUs %d\n", omp_get_num_devices());
   		#ifdef NVIDIA
      			prtAccelInfo();
   		#endif
 	  }  
	#endif

        /* Reading Parameter file */
        read_parameter_file(in.paramfile);

        if ( param.num_threads == 0 )
        {
                fprintf(stderr, "Usage: %d number_of_OMP_Threads \n", param.num_threads);
                exit(1);
        }

      
        for(int ifiles=0; ifiles<param.ndatasets; ifiles++)
        {
              if(global_rank == 0)
              	printf( "\nDataset %d\n", ifiles);

              /*INIT function */
              init(ifiles);

              /* GRIDDING function */
              gridding();

              /* WRITE_GRIDED_DATA function */
              write_gridded_data();

              /* FFTW_DATA function */
              fftw_data();

              /* WRITE_FFTW_DATA function */
              write_fftw_data();
           

              /* WRITE_RESULT function */
              write_result();

              if(global_rank == 0)
              	 printf("*************************************************************\n"); 

       }
       
       //Close MPI Environment
       
       #ifdef USE_MPI
         MPI_Finalize();
       #endif

}
