
apt update && apt install -y git cmake ccache ninja-build tmux

pip install -r requirements.txt

ccache -M 25Gi # Use -M 0 for unlimited
ccache -F 0

export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache

export TORCH_SHOW_CPP_STACKTRACES=1
export DEBUG=1
export REL_WITH_DEB_INFO=1
export USE_DISTRIBUTED=1
export USE_MKLDNN=0
export USE_CUDA=1
export BUILD_TEST=1
export USE_FBGEMM=1
export USE_NNPACK=1
export USE_QNNPACK=1
export USE_XNNPACK=1
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
