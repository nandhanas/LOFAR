#!/bin/bash
#SBATCH -A IscrC_TRACRE
#SBATCH -p m100_usr_prod
### number of nodes
#SBATCH -N 1
### number of hyperthreading threads
#SBATCH --ntasks-per-core=1
### number of MPI tasks per node
#SBATCH --ntasks-per-node=1
### number of openmp threads
#SBATCH --cpus-per-task=1
### number of allocated GPUs per node
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=16000
#SBATCH -t 1:00:00 

export use_cuda=yes
  
if [ "$use_cuda" = "no" ]
then
  export typestring=omp_cpu
  export exe=w-stackingCfftw
fi

if [ "$use_cuda" = "yes" ]
then
  export typestring=cuda
  export exe=w-stackingfftw
fi

export logdir=mpi_${SLURM_NTASKS_PER_NODE}_${typestring}_${SLURM_CPUS_PER_TASK}
echo "Creating $logdir"
rm -fr $logdir
mkdir $logdir

for itest in {1..10}
do
  export logfile=test_${itest}_${logdir}.log
  echo "time mpirun -np $SLURM_NTASKS_PER_NODE /m100/home/userexternal/cgheller/gridding/hpc_imaging/${exe} $SLURM_CPUS_PER_TASK" > $logfile
  time mpirun -np $SLURM_NTASKS_PER_NODE /m100/home/userexternal/cgheller/gridding/hpc_imaging/${exe} $SLURM_CPUS_PER_TASK >> $logfile
  mv $logfile $logdir
  mv timings.dat ${logdir}/timings_${itest}.dat
  cat ${logdir}/timings_all.dat ${logdir}/timings_${itest}.dat >> ${logdir}/timings_all.dat
done

