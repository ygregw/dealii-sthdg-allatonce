# Space-time HDG with Adaptive Mesh Refinement (AMR)

This repository contains [deal.II](https://dealii.org/) codes that implement:

- the space-time hybridizable discontinuous Galerkin method;
- for the advection-diffusion problem;
- on fixed domains;
- using adaptive mesh refinement;
- using the all-at-once approach.

## Demos

Three test cases are implemented:

- A rotating Gaussian pulse test case;<br>
	<img src="https://github.com/ygregw/dealii-sthdg-allatonce/blob/main/misc/rot_pulse_03.png" width="40%">
- A developing boundary layer test case;<br>
	<img src="https://github.com/ygregw/dealii-sthdg-allatonce/blob/main/misc/bnd_layer_final.png" width="40%">
- A developing interior layer test case:<br>
	<img src="https://github.com/ygregw/dealii-sthdg-allatonce/blob/main/misc/int_layer_final.png" width="40%">

Please see **Section 5 - Numerical Examples** of [our paper](https://arxiv.org/abs/2404.04130) for detailed mathematical description as well as convergence history comparison (between uniform and adaptive mesh refinement)
of these test problems.

# Setting up deal.II

GCC14 was released in May 2024 and my personal machine (running
Arch Linux) has since then upgraded from GCC13 to GCC14;

Notes on 2024/05/31: ScaLAPACK v2.2.1 needs to be compiled using GCC13. Since
PETSc will compile and install ScaLAPACK (and then MUMPS) for us, we need to
unify compiler suite for this entire setup.

Arch Linux has packaged
[gcc13](https://archlinux.org/packages/extra/x86_64/gcc13/). Installing this
package makes available the following binaries:
- `gcc-13`
- `g++-13`
- `gfortran-13`

Now we need to compile and install the following softwares (using gcc13):

- Openmpi
- PETSc
	- Hypre
	- ScaLAPACK
	- MUMPS
- p4est
- deal.II

To keep things organized, let's put everything under the same directory `sthdg-amr`:

```shell
$ mkdir sthdg-amr && cd sthdg-amr
$ export BASEDIR=`pwd`
$ mkdir -p $BASEDIR/openmpi $BASEDIR/p4est $BASEDIR/dealii
```

We will let the `git clone` command create the `petsc` directory for us later.

## Installing Openmpi

Useful links:

1. [Open MPI Version 5.0 download page](https://www.open-mpi.org/software/ompi/v5.0/)
2. [Open MPI - Building from source](https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html#building-from-source)

```shell
$ cd $BASEDIR/openmpi
$ wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.1.tar.gz
$ tar xvf openmpi-5.0.1.tar.gz
$ mkdir install
$ cd openmpi-5.0.1
$ ./configure CC=gcc-13 CXX=g++-13 FC=gfortran-13 --prefix=`pwd`/../install 2>&1 | tee config.out
$ make -j4 all 2>&1 | tee make.out
$ make install 2>&1 | tee install.out
```

## Installing PETSc

Useful links:

1. [Interfacing deal.II to PETSc](https://www.dealii.org/current/external-libs/petsc.html)
2. [PETSc's own installation instructions](https://petsc.org/release/install/download/#recommended-obtain-release-version-with-git)
3. [Configuring PETSc](https://petsc.org/release/install/install/)

```shell
$ cd $BASEDIR
$ git clone -b release https://gitlab.com/petsc/petsc.git petsc
$ cd petsc
$ git checkout v3.17.1
$ git describe # you should see "v3.17.1"
$ export PETSC_DIR=`pwd`
$ export PETSC_ARCH=sthdg-amr
./configure --with-shared-libraries=1 --with-x=0 --with-mpi-dir=$BASEDIR/openmpi/install --download-hypre=1 --download-scalapack=1 --download-mumps=1
$ make PETSC_DIR=`pwd` PETSC_ARCH=sthdg-amr all
$ make PETSC_DIR=`pwd` PETSC_ARCH=sthdg-amr check
```

## Installing p4est

Useful links:

1. [Interfacing deal.II to p4est](https://www.dealii.org/developer/external-libs/p4est.html)
2. [p4est's own installation instructions](https://www.p4est.org/)

```shell
$ cd $BASEDIR/p4est
$ wget https://github.com/p4est/p4est.github.io/blob/master/release/p4est-2.8.tar.gz
$ wget https://www.dealii.org/developer/external-libs/p4est-setup.sh
$ chmod u+x p4est-setup.sh
$ mkdir p4est-sthdg
$ ./p4est-setup.sh p4est-2.8.tar.gz `pwd`/p4est-sthdg
```

## Installing deal.II

Useful links:

1. [deal.II's own installation instructions](https://www.dealii.org/9.5.0/readme.html)
2. [Details on the deal.II configuration and build system](https://www.dealii.org/9.5.0/users/cmake_dealii.html)

```shell
$ cd $BASEDIR/dealii
$ wget https://www.dealii.org/downloads/dealii-9.5.2.tar.gz
$ mkdir build install
$ tar xvzf dealii-9.5.2.tar.gz
$ mv dealii-9.5.2 source
$ cd build
$ cmake \
-DCMAKE_C_COMPILER=$BASEDIE/openmpi/install/bin/mpicc \
-DCMAKE_CXX_COMPILER=$BASEDIE/openmpi/install/bin/mpicxx \
-DCMAKE_Fortran_COMPILER=$BASEDIE/openmpi/install/bin/mpif90 \
-DDEAL_II_WITH_MPI=ON \
-DDEAL_II_WITH_PETSC=ON -DPETSC_DIR=$PETSC_DIR -DPETSC_ARCH=sthdg-amr \
-DDEAL_II_WITH_P4EST=ON -DP4EST_DIR=$BASEDIR/p4est/install \
-DCMAKE_INSTALL_PREFIX=`pwd`/../install/ \
../source
$ # after successful configuration
$ make --jobs=N install # specify number of jobs based on specs of your machine
```

# Running the space-time HDG code

Make sure you are in the base directory of this git repo. Then, create `build`
, `build/vtus_uniform` and `build/vtus_adaptive` directories. The `vtus_*`
directories stores the VTK output files (to be visualized by softwares like
ParaView or VisIt).

```shell
$ basename $PWD # you should see "dealii-sthdg-allatonce"
$ mkdir -p build/vtus_uniform build/vtus_adaptive
```

Now configure and compile the code.

```shell
$ cd build
$ cmake ..
$ make
```

When successful, you should obtain executable `sthdg-advdif-allatonce`. It
takes five commandline options:

1. `-n N`: sets diffusion parameter to be 10^{-N};
2. `-c N`: sets N uniform refinement cycles;
3. `-p N`: uses finite elements of polynomial degree N;
4. `-a`: toggles amr mode on;
5. `-o`: toggles vtu output on.

## Example runs

**Boundary layer problem**:\
After compiling with the boundary layer problem, we run with 10^{-3} being the
diffusion parameter, 4 mpi processes, linear polynomial as the finite elements
and with AMR.

```shell
$ mpiexec -n 4 ./sthdg-advdif-allatonce -n 3 -c 4 -p 1 -a | tee bnd_n3c4p1.txt
================================================================================
START DATE: 2024/6/1, TIME: 22:19:26
--------------------------------------------------------------------------------
Boundary Layer Problem, nu = 0.001
Running with 4 MPI processes, PETSc sparse direct MUMPS solver
Finite element space: FE_FaceQ<3>(1), FE_DGQ<3>(1)
Space-time IP-HDG, with semi-centered-flux penalty
================================================================================
--------------------------------------------------------------------------------
Cycle 1
--------------------------------------------------------------------------------
Set up system...
  Global mesh: 	1000 cells
  DoFHandler: 	13200 DoFs
  Mem usage: 	283 MB
  Done! (5.9094s)
Output results...
  Triple norm error: 	0.311306
  Estimated error: 	6.25136
--------------------------------------------------------------------------------
Cycle 2
--------------------------------------------------------------------------------
Set up system...
  Global mesh: 	2750 cells
  DoFHandler: 	37708 DoFs
  Mem usage: 	319 MB
  Done! (14.294s)
Output results...
  Triple norm error: 	0.291142
  Estimated error: 	4.94678
--------------------------------------------------------------------------------
Cycle 3
--------------------------------------------------------------------------------
Set up system...
  Global mesh: 	7566 cells
  DoFHandler: 	107232 DoFs
  Mem usage: 	368 MB
  Done! (41.1466s)
Output results...
  Triple norm error: 	0.260768
  Estimated error: 	3.77887
--------------------------------------------------------------------------------
Cycle 4
--------------------------------------------------------------------------------
Set up system...
  Global mesh: 	20663 cells
  DoFHandler: 	296260 DoFs
  Mem usage: 	470 MB
  Done! (145.156s)
Output results...
  Triple norm error: 	0.244448
  Estimated error: 	2.84276
==================================================================================================================================
Convergence History:
----------------------------------------------------------------------------------------------------------------------------------
cells  dofs       L2            sH1           tH1          dfjp          adjp         neum          nojp         tnorm       est     eff   efft
 1000  13200 4.0e-03     - 4.1e-03     - 1.0e-04     - 3.1e-02     - 3.1e-01    - 6.2e-03     - 8.4e-03     - 3.1e-01    - 6.3e+00 7.4e+02 20.1
 2750  37708 5.2e-03 -0.80 9.5e-03 -2.44 8.0e-05  0.69 4.1e-02 -0.78 2.9e-01 0.21 8.1e-03 -0.78 1.4e-02 -1.37 2.9e-01 0.19 4.9e+00 3.6e+02 17.0
 7566 107232 5.5e-03 -0.13 4.5e-02 -4.44 7.1e-05  0.35 5.0e-02 -0.61 2.5e-01 0.39 8.5e-03 -0.12 4.6e-02 -3.50 2.6e-01 0.32 3.8e+00 8.2e+01 14.5
20663 296260 4.6e-03  0.54 1.3e-01 -3.12 1.0e-04 -1.01 5.6e-02 -0.31 2.0e-01 0.68 7.0e-03  0.56 1.3e-01 -3.06 2.4e-01 0.19 2.8e+00 2.2e+01 11.6
==================================================================================================================================


+---------------------------------------------+------------+------------+
| Total wallclock time elapsed since start    |       218s |            |
|                                             |            |            |
| Section                         | no. calls |  wall time | % of total |
+---------------------------------+-----------+------------+------------+
| assemble_system                 |         8 |      65.9s |        30% |
| init_mesh                       |         1 |       0.9s |      0.41% |
| refine_mesh                     |         3 |      10.1s |       4.6% |
| setup_system                    |         4 |      16.6s |       7.6% |
| solve                           |         4 |       129s |        59% |
+---------------------------------+-----------+------------+------------+
```

**Interior layer problem**:\
After compiling with the interior layer problem, we run with 10^{-3} being the
diffusion parameter, 4 mpi processes, linear polynomial as the finite elements
and with AMR:

```shell
$ mpiexec -n 4 ./sthdg-advdif-allatonce -n 3 -c 4 -p 1 -a | tee int_n3c4p1.txt
================================================================================
START DATE: 2024/6/1, TIME: 22:12:18
--------------------------------------------------------------------------------
Interior Layer Problem, nu = 0.001
Running with 4 MPI processes, PETSc sparse direct MUMPS solver
Finite element space: FE_FaceQ<3>(1), FE_DGQ<3>(1)
Space-time IP-HDG, with semi-centered-flux penalty
================================================================================
--------------------------------------------------------------------------------
Cycle 1
--------------------------------------------------------------------------------
Set up system...
  Global mesh: 	1000 cells
  DoFHandler: 	13200 DoFs
  Mem usage: 	276 MB
  Done! (5.89829s)
Output results...
  Triple norm error: 	2.03028
  Estimated error: 	2.12749
--------------------------------------------------------------------------------
Cycle 2
--------------------------------------------------------------------------------
Set up system...
  Global mesh: 	2743 cells
  DoFHandler: 	37636 DoFs
  Mem usage: 	323 MB
  Done! (14.6107s)
Output results...
  Triple norm error: 	1.4577
  Estimated error: 	1.48951
--------------------------------------------------------------------------------
Cycle 3
--------------------------------------------------------------------------------
Set up system...
  Global mesh: 	7944 cells
  DoFHandler: 	114364 DoFs
  Mem usage: 	372 MB
  Done! (51.2688s)
Output results...
  Triple norm error: 	1.24468
  Estimated error: 	1.30175
--------------------------------------------------------------------------------
Cycle 4
--------------------------------------------------------------------------------
Set up system...
  Global mesh: 	23169 cells
  DoFHandler: 	344232 DoFs
  Mem usage: 	510 MB
  Done! (236.094s)
Output results...
  Triple norm error: 	0.995676
  Estimated error: 	1.10106
==================================================================================================================================
Convergence History:
----------------------------------------------------------------------------------------------------------------------------------
cells  dofs       L2          sH1          tH1          dfjp          adjp         neum         nojp        tnorm       est     eff   efft
 1000  13200 1.0e-01    - 2.0e+00    - 1.9e-03    - 9.9e-03     - 9.9e-02    - 1.7e-01    - 2.0e+00    - 2.0e+00    - 2.1e+00 1.0e+00  1.0
 2743  37636 9.2e-02 0.38 1.4e+00 0.95 1.4e-03 0.88 9.9e-03 -0.02 7.3e-02 0.88 1.5e-01 0.49 1.5e+00 0.95 1.5e+00 0.95 1.5e+00 1.0e+00  1.0
 7944 114364 7.8e-02 0.44 1.2e+00 0.43 1.0e-03 0.87 1.2e-02 -0.42 6.4e-02 0.36 1.2e-01 0.49 1.2e+00 0.43 1.2e+00 0.43 1.3e+00 1.0e+00  1.0
23169 344232 6.9e-02 0.33 9.9e-01 0.61 6.9e-04 1.04 1.2e-02 -0.12 5.0e-02 0.65 1.1e-01 0.28 9.9e-01 0.61 1.0e+00 0.61 1.1e+00 1.1e+00  1.1
==================================================================================================================================


+---------------------------------------------+------------+------------+
| Total wallclock time elapsed since start    |       320s |            |
|                                             |            |            |
| Section                         | no. calls |  wall time | % of total |
+---------------------------------+-----------+------------+------------+
| assemble_system                 |         8 |        73s |        23% |
| init_mesh                       |         1 |      0.83s |      0.26% |
| refine_mesh                     |         3 |      11.3s |       3.5% |
| setup_system                    |         4 |      20.6s |       6.4% |
| solve                           |         4 |       220s |        69% |
+---------------------------------+-----------+------------+------------+
```
