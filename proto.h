/* function declaration */

//Revamped by NANDHANA SAKTHIVEL as part of her Master thesis, DSSC, UNITS, Italy

/* init.c */

void init(int i);
void op_filename();
void read_parameter_file(char *);
void fileName(char datapath[900], char file[30]);
void readMetaData(char fileLocal[1000]);
void metaData_calculation();
void allocate_memory();
void readData();


/*  gridding.c */

void gridding();
void initialize_array();
void gridding_data();
void write_gridded_data();
void copy_win_ptrs( void ***, win_t *, int n );

/* fourier_transform.c */

void fftw_data();
void write_fftw_data();
void write_result();

/* reduce.c */

int reduce_ring (int );
int reduce_binomial (int );
int shmem_reduce_ring  ( int, int, int_t, map_t *, double * restrict, blocks_t *);
int shmem_reduce_binomial( int, int, int, map_t *, double * restrict, int );

