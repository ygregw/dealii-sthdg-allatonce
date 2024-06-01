# Space-time HDG with Adaptive Mesh Refinement (AMR)

This repository contains [deal.II](https://dealii.org/) codes that implement:

- the space-time hybridizable discontinuous Galerkin method;
- for the advection-diffusion problem;
- on fixed domains;
- using adaptive mesh refinement;
- using the all-at-once approach.

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
