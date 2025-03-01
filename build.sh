export DEBUG=1
export USE_DEBUG=1
export USE_CUDA=1
export BUILD_TEST=0
export USE_FLASH_ATTENTION=0
export MAX_JOBS=12


# Add this at the root cmake
# add_subdirectory(demo)

python setup.py develop
pushd torch/lib
ln -sf ../../build/lib/libtorch_cpu.* .
popd


# cmake --build . --target install --config Debug -- -j 6