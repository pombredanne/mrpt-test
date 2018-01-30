## Simple test for MRPT approximate nearest neighbor search

## Pipeline


1. Run `./install_eigen.sh`: installs eigen linear algebra library; current version is [`Eigen 3.3.4.`](http://eigen.tuxfamily.org/index.php?title=Main_Page)

2. Install a current version of MRPT: `git clone https://github.com/teemupitkanen/mrpt.git`, and
specify the installation directory to the variable `MRPT_PATH` in `cpp/Makefile` (default is the directory that contains this repo).

3. `make` the test file in `cpp` folder and run it using `cpp/test`. 
