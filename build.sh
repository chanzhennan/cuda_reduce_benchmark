rm -rf build
mkdir build && cd build 
cmake ..
make -j
./cuda_benchmark