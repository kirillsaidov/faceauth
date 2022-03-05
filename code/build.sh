#! /bin/sh

cd build;

cmake -DCMAKE_PREFIX_PATH=libtorch -S .. -B . --config Release; 

make




