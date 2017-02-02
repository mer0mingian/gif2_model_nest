Installation instructions:

1.) change directory to the path containing the 'gif2_module' folder.
2.) mkdir gif2_module_build
3.) cd gif2_module_build
4.) cmake ../gif2_module
5.) make
6.) make install
7.) to make a session-wise installation in Nest, type: /gif2_module Install

Requirements:
- cmake
- 'which nest' points to nest installation path

Created by Daniel Mingers, Jun 2016, email: mer0.dm0@gmail.com
Based on "From Subthreshold Resonance to Firing-Rate Resonance";
M.J.E. Richardson, N. Brunel, V. Hakim; J Neurophysiol. 89: 2538-2554 (2003)


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/d.mingers/opt/nest.install/lib64/nest/

# fetch the current master branch from git

load modules: autotools, pystuff_new, openmpi

cd bzw mkdir--> build_dir: /home/mer0/nest/nest_build
 cmake ../../gitdata/nest-dev/ 
-DCMAKE_INSTALL_PREFIX=$HOME/nest/nest_install 
# -Dwith-python=/home/mer0/anaconda2/bin/python 
-DPYTHON_LIBRARY=/home/mer0/anaconda2/lib/libpython2.7.so 
-DPYTHON_INCLUDE_DIR=/home/mer0/anaconda2/include/python2.7
-Dwith-openmp=-fopenmp 
-Dwith-mpi=/opt/software/mpi/openmpi/default/ 

#optionally include: -Dexternal_modules=gif2
make -j4
make install

make installcheck

--> .bashprofile
# NEST directory
source /home/d.mingers/opt/nest.install/bin/nest_vars.sh
export NEST_INSTALL_DIR=/home/d.mingers/opt/nest.install/

--> .nestrc
replace the mpi section by this:
  /mpirun
  [/integertype /stringtype /stringtype]
  [/numproc     /executable /scriptfile]
  {
   () [
    (mpirun -np ) numproc cvs ( ) executable ( ) scriptfile
   ] {join} Fold
  } Function def


cd-> MyModule_builddir
# check that the current version of cmake_lists.txt is identical to the one in the example of the master branch
cmake3 ../MyModule
make
make install

nest
/mymodule Install

# zu beachten:
"which nest" sollte auf das install-verzeichnis zeigen



# working cmake command for blaustein with all modules loaded:
cmake3 ../nest.gitclone/ -DCMAKE_INSTALL_PREFIX=$HOME/opt/nest.install 
-DPYTHON_LIBRARY=/opt/software/pystuff_new/lib/libpython2.7.so 
-DPYTHON_INCLUDE_DIR=/opt/software/pystuff_new/include/python2.7 
-Dwith-openmp=-fopenmp 
-Dwith-mpi=/opt/software/mpi/openmpi/default/