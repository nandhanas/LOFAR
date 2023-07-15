#ifndef W_PROJECT_H_
#define W_PROJECT_H_

#define NWORKERS -1    //100
#define NTHREADS 32
#define PI 3.14159265359
#define REAL_TYPE double
#ifdef __CUDACC__
extern "C"
#endif

void wstack(
     int,
     long,
     long,
     long,
     double*,
     double*,
     double*,
     float*,
     float*,
     float*,
     double,
     double,
     int,
     int,
     int,
     double*,
     int);

#ifdef __CUDACC__
extern "C"
#endif
int test(int nnn);

#ifdef __CUDACC__
extern "C"
#endif
void phase_correction(
     double*,
     double*,
     double*,
     int,
     int,
     int,
     int,
     int,
     double,
     double,
     double,
     int);
#endif 
