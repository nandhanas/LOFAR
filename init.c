#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "allvars.h"
#include "proto.h"

//Revamped by NANDHANA SAKTHIVEL as part of her Master thesis, DSSC, UNITS, Italy


void init(int index)
{

  TAKE_TIME_START(total);

   // DAV: the corresponding KernelLen is calculated within the wstack function. It can be anyway hardcoded for optimization
   dx = 1.0/(double)param.grid_size_x;
   dw = 1.0/(double)param.num_w_planes;
   w_supporth = (double)((param.w_support-1)/2)*dx;
                            
   // MESH SIZE
   int local_grid_size_x;// = 8;
   int local_grid_size_y;// = 8;
   
   // set the local size of the image
   local_grid_size_x = param.grid_size_x;
   nsectors = NUM_OF_SECTORS;
   if (nsectors < 0) nsectors = size;
   local_grid_size_y = param.grid_size_y/nsectors;
   //nsectors = size;

   // LOCAL grid size
   xaxis = local_grid_size_x;
   yaxis = local_grid_size_y;

   #ifdef USE_MPI
   #ifdef ONE_SIDE
   numa_init( global_rank, size, &MYMPI_COMM_WORLD, &Me );
   numa_expose(&Me,0);
   #endif
   #endif

   TAKE_TIME_START(setup);

   // INPUT FILES (only the first ndatasets entries are used)
   strcpy(datapath,param.datapath_multi[index]);
   sprintf(num_buf, "%d", index);
   
   //Changing the output file names
   op_filename();
   
   // Read metadata
   fileName(datapath, in.metafile);
   readMetaData(filename);
   
   // Local Calculation
   metaData_calculation();
   
   // Allocate Data Buffer
   allocate_memory();
   
   // Reading Data
   readData();
   
  #ifdef USE_MPI
   MPI_Barrier(MPI_COMM_WORLD);
  #endif
   
   TAKE_TIME_STOP(setup);
}

void op_filename() {

   if(global_rank == 0)
   {   
   	strcpy(buf, num_buf);
   	strcat(buf, outparam.outfile);
   	strcpy(out.outfile, buf);
   
   	strcpy(buf, num_buf);
   	strcat(buf, outparam.outfile1);
   	strcpy(out.outfile1, buf); 

   	strcpy(buf, num_buf);
   	strcat(buf, outparam.outfile2);
   	strcpy(out.outfile2, buf);

   	strcpy(buf, num_buf);
   	strcat(buf, outparam.outfile3);
   	strcpy(out.outfile3, buf);

   	strcpy(buf, num_buf);
   	strcat(buf, outparam.fftfile);
   	strcpy(out.fftfile, buf);

   	strcpy(buf, num_buf);
   	strcat(buf, outparam.fftfile2);
   	strcpy(out.fftfile2, buf);

   	strcpy(buf, num_buf);
   	strcat(buf, outparam.fftfile3);
   	strcpy(out.fftfile3, buf);
    
   	strcpy(buf, num_buf);
   	strcat(buf, outparam.logfile);
   	strcpy(out.logfile, buf);

   	strcpy(buf, num_buf);
   	strcat(buf, outparam.extension);
   	strcpy(out.extension, buf);

   	strcpy(buf, num_buf);
   	strcat(buf, outparam.timingfile);
   	strcpy(out.timingfile, buf);
    }

   /* Communicating the relevent parameters to the other process */
  #ifdef USE_MPI
   MPI_Bcast(&out, sizeof(struct op), MPI_BYTE, 0, MPI_COMM_WORLD);
  #endif

   return;
}

void read_parameter_file(char *fname)
{
   if(global_rank == 0)
   {
     if( (file.pFile = fopen (fname,"r")) != NULL )
   	{
	  char buf1[30], buf2[100], buf3[30], num[30];
	  int i = 1;
	  while(fscanf(file.pFile, "%s" "%s", buf1, buf2) != EOF)
	    {
	      if(strcmp(buf1, "num_threads") == 0)
		{
		  param.num_threads = atoi(buf2);
		}
	      if(strcmp(buf1, "Datapath1") == 0)
		{
		  strcpy(param.datapath_multi[0], buf2);
		  i++;
		}
	      if(strcmp(buf1, "ndatasets") == 0)
		{
		  param.ndatasets = atoi(buf2);
		}
	      if(strcmp(buf1, "w_support") == 0)
		{
		  param.w_support = atoi(buf2);
		}
	      if(strcmp(buf1, "grid_size_x") == 0)
		{
		  param.grid_size_x = atoi(buf2);
		}
	      if(strcmp(buf1, "grid_size_y") == 0)
		{
		  param.grid_size_y = atoi(buf2);
		}
	      if(strcmp(buf1, "num_w_planes") == 0)
		{
		  param.num_w_planes = atoi(buf2);
		}
	      if(strcmp(buf1, "ufile") == 0)
		{
		  strcpy(in.ufile, buf2);
		}
	      if(strcmp(buf1, "vfile") == 0)
		{
		  strcpy(in.vfile, buf2);
		}
	      if(strcmp(buf1, "wfile") == 0)
		{
		  strcpy(in.wfile, buf2);
		}
	      if(strcmp(buf1, "weightsfile") == 0)
		{
		  strcpy(in.weightsfile, buf2);
		}
	      if(strcmp(buf1, "visrealfile") == 0)
		{
		  strcpy(in.visrealfile, buf2);
		}
	      if(strcmp(buf1, "visimgfile") == 0)
		{
		  strcpy(in.visimgfile, buf2);
		}
	      if(strcmp(buf1, "metafile") == 0)
		{
		  strcpy(in.metafile, buf2);
		}
	      if(strcmp(buf1, "outfile") == 0)
		{
		  strcpy(outparam.outfile, buf2);
		}
	      if(strcmp(buf1, "outfile1") == 0)
		{
		  strcpy(outparam.outfile1, buf2);
		}
	      if(strcmp(buf1, "outfile2") == 0)
		{
		  strcpy(outparam.outfile2, buf2);
		}
	      if(strcmp(buf1, "outfile3") == 0)
		{
		  strcpy(outparam.outfile3, buf2);
		}
	      if(strcmp(buf1, "fftfile") == 0)
		{
		  strcpy(outparam.fftfile, buf2);
		}
	      if(strcmp(buf1, "fftfile2") == 0)
		{
		  strcpy(outparam.fftfile2, buf2);
		}
	      if(strcmp(buf1, "fftfile3") == 0)
		{
		  strcpy(outparam.fftfile3, buf2);
		}
	      if(strcmp(buf1, "logfile") == 0)
		{
		  strcpy(outparam.logfile, buf2);
		}
	      if(strcmp(buf1, "extension") == 0)
		{
		  strcpy(outparam.extension, buf2);
		}
	      if(strcmp(buf1, "timingfile") == 0)
		{
		  strcpy(outparam.timingfile, buf2);
		}
	      if(strcmp(buf1, "verbose_level") == 0)
		{
		  verbose_level = atoi(buf1);
		}

	      if(param.ndatasets > 1)
		{
                   
		  sprintf(num, "%d", i);
		  strcat(strcpy(buf3,"Datapath"),num);
		  if(strcmp(buf1,buf3) == 0)
		    {
		      strcpy(param.datapath_multi[i-1], buf2);
		      i++;
		    } 
		}
	    }
	  fclose(file.pFile);
      
       }
       else
       {
     		printf("error opening paramfile");
     		exit(1);
       }
    }
   
    /* Communicating the relevent parameters to the other process */

  #ifdef USE_MPI
   double twt;
   TAKE_TIMEwt(twt);
   MPI_Bcast(&in, sizeof(struct ip), MPI_BYTE, 0, MPI_COMM_WORLD);
   MPI_Bcast(&outparam, sizeof(struct op), MPI_BYTE, 0, MPI_COMM_WORLD);
   MPI_Bcast(&param, sizeof(struct parameter), MPI_BYTE, 0, MPI_COMM_WORLD);
   ADD_TIMEwt(mpi, twt);
  #endif

}


void fileName(char datapath[900], char file[30]) {
     strcpy(filename,datapath);
     strcat(filename,file);
}


void readMetaData(char fileLocal[1000]) {

  if(global_rank == 0) 
    {
      if( (file.pFile = fopen (fileLocal,"r")) != NULL )
        {
	  int items = 0;
	  items += fscanf(file.pFile,"%ld",&metaData.Nmeasures);
	  items += fscanf(file.pFile,"%ld",&metaData.Nvis);
	  items += fscanf(file.pFile,"%ld",&metaData.freq_per_chan);
	  items += fscanf(file.pFile,"%ld",&metaData.polarisations);
	  items += fscanf(file.pFile,"%ld",&metaData.Ntimes);
	  items += fscanf(file.pFile,"%lf",&metaData.dt);
	  items += fscanf(file.pFile,"%lf",&metaData.thours);
	  items += fscanf(file.pFile,"%ld",&metaData.baselines);
	  items += fscanf(file.pFile,"%lf",&metaData.uvmin);
	  items += fscanf(file.pFile,"%lf",&metaData.uvmax);
	  items += fscanf(file.pFile,"%lf",&metaData.wmin);
	  items += fscanf(file.pFile,"%lf",&metaData.wmax);
	  
	  fclose(file.pFile);
        } 
      else
	{
	  printf("error opening meta file %s\n",
		 fileLocal);
	  exit(1);
	}
    }
      
  /* Communicating the relevent parameters to the other process */
 #ifdef USE_MPI
  MPI_Bcast(&metaData, sizeof(struct meta), MPI_BYTE, 0, MPI_COMM_WORLD);
 #endif
  
}

void metaData_calculation() {
   
     int nsub = 1000;
     if( global_rank == 0 )
       printf("Subtracting last %d measurements\n",nsub);
     metaData.Nmeasures = metaData.Nmeasures-nsub;
     metaData.Nvis = metaData.Nmeasures*metaData.freq_per_chan*metaData.polarisations;

     // calculate the coordinates of the center
     #warning why it it not used?
     double uvshift = metaData.uvmin/(metaData.uvmax-metaData.uvmin);

     if (global_rank == 0)
       {
	 printf("N. measurements %ld\n",metaData.Nmeasures);
	 printf("N. visibilities %ld\n",metaData.Nvis);
       }
     
     // Set temporary local size of points
     long nm_pe = (long)(metaData.Nmeasures/size);
     long remaining = metaData.Nmeasures%size;

     startrow = global_rank*nm_pe;
     if (global_rank == size-1)nm_pe = nm_pe+remaining;

     metaData.Nmeasures = nm_pe;
     metaData.Nvis = metaData.Nmeasures*metaData.freq_per_chan*metaData.polarisations;
     metaData.Nweights = metaData.Nmeasures*metaData.polarisations;

    #ifdef VERBOSE
     printf("N. measurements on %d %ld\n",global_rank,metaData.Nmeasures);
     printf("N. visibilities on %d %ld\n",global_rank,metaData.Nvis);
    #endif

}

void allocate_memory() {


     // DAV: all these arrays can be allocatate statically for the sake of optimization. However be careful that if MPI is used
     // all the sizes are rescaled by the number of MPI tasks
     //  Allocate arrays
     
     data.uu = (double*) calloc(metaData.Nmeasures,sizeof(double));
     data.vv = (double*) calloc(metaData.Nmeasures,sizeof(double));
     data.ww = (double*) calloc(metaData.Nmeasures,sizeof(double));
     data.weights = (float*) calloc(metaData.Nweights,sizeof(float));
     data.visreal = (float*) calloc(metaData.Nvis,sizeof(float));
     data.visimg = (float*) calloc(metaData.Nvis,sizeof(float));


     // Create sector grid
     
     size_of_grid = 2*param.num_w_planes*xaxis*yaxis;
     gridss       = (double*) calloc(size_of_grid,sizeof(double));
     gridss_w     = (double*) calloc(size_of_grid,sizeof(double));
     gridss_real  = (double*) calloc(size_of_grid/2,sizeof(double));
     gridss_img   = (double*) calloc(size_of_grid/2,sizeof(double));
   
     #ifdef USE_MPI   
     #ifdef ONE_SIDE
     numa_allocate_shared_windows( &Me, size_of_grid*sizeof(double)*1.1, sizeof(double)*1.1 );
     #endif
     #endif
 
     // Create destination slab
      grid = (double*) calloc(size_of_grid,sizeof(double));
     
     // Create temporary global grid
     #ifndef USE_MPI
     	   double * gridtot = (double*) calloc(2*grid_size_x*grid_size_y*num_w_planes,sizeof(double));
     #endif

}

void readData() {

     if(global_rank == 0) 
     {

          printf("READING DATA\n");

     }

     fileName(datapath, in.ufile);
     if( (file.pFile = fopen (filename,"rb")) != NULL )
     {
          fseek (file.pFile,startrow*sizeof(double),SEEK_SET);
          size_t res = fread(data.uu, sizeof(double), metaData.Nmeasures, file.pFile);
	  if( res != metaData.Nmeasures )
	    printf("an error occurred while reading file %s\n", filename);
          fclose(file.pFile);
     }
     else
       {
	 printf("error opening ucoord file %s\n",
		filename );
	 exit(1);
       }

     fileName(datapath, in.vfile);
     if( (file.pFile = fopen (filename,"rb")) != NULL )
     {
          fseek (file.pFile,startrow*sizeof(double),SEEK_SET);
          size_t res = fread(data.vv, sizeof(double), metaData.Nmeasures, file.pFile);
	  if( res != metaData.Nmeasures )
	    printf("an error occurred while reading file %s\n", filename);
	      
          fclose(file.pFile);
     }
     else
     {
       printf("error opening vcoord file %s\n",
	      filename);
       exit(1);
     }

     fileName(datapath, in.wfile);
     if( (file.pFile = fopen (filename,"rb")) != NULL )
     {
          fseek (file.pFile,startrow*sizeof(double),SEEK_SET);
          size_t res = fread(data.ww,sizeof(double), metaData.Nmeasures, file.pFile);
	  if( res != metaData.Nmeasures )
	    printf("an error occurred while reading file %s\n", filename);	      
          fclose(file.pFile);
     }
     else
     {
       printf("error opening wcoord file %s\n",
	      filename);
          exit(1);
     }

     fileName(datapath, in.weightsfile);
     if( (file.pFile = fopen (filename,"rb")) != NULL)
     {
          fseek (file.pFile,startrow*metaData.polarisations*sizeof(float),SEEK_SET);
          size_t res = fread(data.weights, sizeof(float), metaData.Nweights, file.pFile);
	  if( res != metaData.Nweights )
	    printf("an error occurred while reading file %s\n", filename);
          fclose(file.pFile);
     }
     else
     {
       printf("error opening weights file %s\n",
	      filename);
          exit(1);
     }

     fileName(datapath, in.visrealfile);
     if((file.pFile = fopen (filename,"rb")) != NULL )
     {
          fseek (file.pFile,startrow*metaData.freq_per_chan*metaData.polarisations*sizeof(float),SEEK_SET);
          size_t res = fread(data.visreal, sizeof(float), metaData.Nvis, file.pFile);
	  if( res != metaData.Nvis )
	    printf("an error occurred while reading file %s\n", filename);
          fclose(file.pFile);
     }
     else
     {
       printf("error opening visibilities_real file %s\n",
	      filename);
          exit(1);
     }

     fileName(datapath, in.visimgfile);
     if( (file.pFile = fopen (filename,"rb")) != NULL )
     {
          fseek (file.pFile,startrow*metaData.freq_per_chan*metaData.polarisations*sizeof(float),SEEK_SET);
          size_t res = fread(data.visimg, sizeof(float), metaData.Nvis, file.pFile);
	  if( res != metaData.Nvis )
	    printf("an error occurred while reading file %s\n", filename);
          fclose(file.pFile);
     }
     else
     {
       printf("error opening visibilities_img file %s\n",
	      filename);
          exit(1);
     }
     

     #ifdef USE_MPI
          MPI_Barrier(MPI_COMM_WORLD);
     #endif

}
