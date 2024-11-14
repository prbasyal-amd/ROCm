#! /usr/bin/bash

set -ex

apt-get update 
DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true apt-get install --no-install-recommends -y $(grep -v '^#' /tmp/packages)
apt-get clean 
rm -rf /var/cache/apt/ /var/lib/apt/lists/* /etc/apt/apt.conf.d/01proxy

#Install git 2.17.1 version
curl -o git.tar.gz https://cdn.kernel.org/pub/software/scm/git/git-2.17.1.tar.gz 
tar -zxf git.tar.gz
cd git-* 
make prefix=/usr/local all 
make prefix=/usr/local install
git --version

#install argparse and CppHeaderParser python modules for roctracer and rocprofiler
# CppHeader needs setuptools. setuptools needs wheel.
pip3 install --no-cache-dir setuptools wheel tox
pip3 install --no-cache-dir CppHeaderParser argparse requests lxml barectf recommonmark jinja2==3.0.0  websockets matplotlib numpy scipy minimal msgpack pytest sphinx joblib cmake==3.25.2 pandas myst-parser

# Allow sudo for everyone user
echo 'ALL ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/everyone

# Install OCaml packages to build LLVM's OCaml bindings to be used in lightning compiler test pipeline
wget -nv https://sourceforge.net/projects/opam.mirror/files/2.1.4/opam-2.1.4-x86_64-linux -O /usr/local/bin/opam 
chmod +x /usr/local/bin/opam 
opam init --yes --disable-sandboxing 
opam install ctypes --yes

# Install and modify git-repo (#!/usr/bin/env python -> #!/usr/bin/env python3)
curl https://storage.googleapis.com/git-repo-downloads/repo > /usr/bin/repo
chmod a+x /usr/bin/repo

# Build a newer version of doxygen because of issue 6456
#Install doxygen 1.8.11 for dbgapi
mkdir /tmp/doxygen
cd /tmp/doxygen
curl -LO https://sourceforge.net/projects/doxygen/files/rel-1.8.18/doxygen-1.8.18.src.tar.gz 
tar xzf doxygen-1.8.18.src.tar.gz 
cd doxygen-1.8.18 
mkdir -p build 
cd build 
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -Wno-dev .. 
make -j install 
cd ../.. 
rm -rf /tmp/doxygen

# Build ccache from the source
cd /tmp 
git clone https://github.com/ccache/ccache -b v4.7.5 
cd ccache 
mkdir build 
cd build 
cmake -DCMAKE_BUILD_TYPE=Release .. 
make 
make install 
cd /tmp 
rm -rf ccache

# Install sharp from MLNX_OFED_LINUX as dependency for rccl-rdma-sharp-plugins
cd /var/tmp
mkdir mlnx 
wget -O mlnx/tar.tgz https://content.mellanox.com/ofed/MLNX_OFED-5.2-1.0.4.0/MLNX_OFED_LINUX-5.2-1.0.4.0-ubuntu20.04-x86_64.tgz 
tar -xz -C mlnx -f mlnx/tar.tgz 
apt-key add mlnx/*/RPM-GPG-KEY-Mellanox 
echo "deb [arch=amd64] file:$(echo $PWD/mlnx/*/DEBS) ./" > /etc/apt/sources.list.d/sharp.list 
apt update 
apt install -y sharp 
apt clean 
rm -rf /var/cache/apt/ /var/lib/apt/lists/* mlnx /etc/apt/sources.list.d/sharp.list

#Install older version of hwloc-devel package for rocrtst
curl -lO https://download.open-mpi.org/release/hwloc/v1.11/hwloc-1.11.13.tar.bz2 
tar -xvf hwloc-1.11.13.tar.bz2 
cd hwloc-1.11.13 
./configure 
make 
make install 
cp /usr/local/lib/libhwloc.so.5 /usr/lib 
hwloc-info --version

## Install googletest from source
mkdir -p /tmp/gtest 
cd /tmp/gtest 
wget https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip -O googletest.zip 
unzip googletest.zip 
cd googletest-1.14.0/
mkdir build
cd build 
cmake ..
make -j$(nproc)
make install
rm -rf /tmp/gtest

## Install gRPC from source
## RDC Pre-requisites
GRPC_ARCHIVE=grpc-1.61.0.tar.gz
mkdir /tmp/grpc
mkdir /usr/grpc
cd /tmp 
git clone --recurse-submodules -b v1.61.0 https://github.com/grpc/grpc 
cd grpc
mkdir -p cmake/build
cd cmake/build
cmake -DgRPC_INSTALL=ON -DBUILD_SHARED_LIBS=ON -DgRPC_BUILD_TESTS=OFF  -DCMAKE_INSTALL_PREFIX=/usr/grpc   -DCMAKE_CXX_STANDARD=14  -DCMAKE_SHARED_LINKER_FLAGS_INIT=-Wl,--enable-new-dtags,--build-id=sha1,--rpath,'$ORIGIN' ../.. 
make -j$(nproc)
make install 
cd  / 
rm -rf /tmp/grpc 

## Download prebuilt AMD multithreaded blis (2.0)
## Reference : https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/install.sh#L403
mkdir -p /tmp/blis
cd /tmp/blis 
wget -O - https://github.com/amd/blis/releases/download/2.0/aocl-blis-mt-ubuntu-2.0.tar.gz | tar xfz - 
mv amd-blis-mt /usr/blis 
cd / 
rm -rf /tmp/blis

## Download aocl-linux-gcc-4.2.0_1_amd64.deb
mkdir -p /tmp/aocl 
cd /tmp/aocl 
wget -nv https://download.amd.com/developer/eula/aocl/aocl-4-2/aocl-linux-gcc-4.2.0_1_amd64.deb 
apt install ./aocl-linux-gcc-4.2.0_1_amd64.deb
rm -rf /tmp/aocl

## lapack(3.9.1v)
## Reference https://github.com/ROCmSoftwarePlatform/rocSOLVER/blob/develop/install.sh#L174
lapack_version=3.9.1
lapack_srcdir=lapack-$lapack_version
lapack_blddir=lapack-$lapack_version-bld
mkdir -p /tmp/lapack
cd /tmp/lapack
mkdir -p /tmp/lapack
cd /tmp/lapack
rm -rf "$lapack_srcdir" "$lapack_blddir" 
wget -O - https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.9.1.tar.gz | tar xzf - 
cmake -H$lapack_srcdir -B$lapack_blddir -DCMAKE_BUILD_TYPE=Release  -DCMAKE_Fortran_FLAGS=-fno-optimize-sibling-calls  -DBUILD_TESTING=OFF  -DCBLAS=ON  -DLAPACKE=OFF
make -j$(nproc) -C "$lapack_blddir"
make -C "$lapack_blddir" install
cd $lapack_blddir
cp -r ./include/* /usr/local/include/ 
cp -r ./lib/* /usr/local/lib 
cd / 
rm -rf /tmp/lapack

## FMT(7.1.3v)
## Reference https://github.com/ROCmSoftwarePlatform/rocSOLVER/blob/develop/install.sh#L152
fmt_version=7.1.3 
fmt_srcdir=fmt-$fmt_version
fmt_blddir=fmt-$fmt_version-bld
rm -rf "$fmt_srcdir" "$fmt_blddir"
wget -O - https://github.com/fmtlib/fmt/archive/refs/tags/$fmt_version.tar.gz  | tar xzf -
cmake -H$fmt_srcdir -B$fmt_blddir -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_CXX_STANDARD_REQUIRED=ON -DFMT_DOC=OFF -DFMT_TEST=OFF 
make -j$(nproc) -C "$fmt_blddir" 
make -C "$fmt_blddir" install

# Build and install libjpeg-turbo
mkdir -p /tmp/libjpeg-turbo 
cd /tmp/libjpeg-turbo
wget -nv https://github.com/rrawther/libjpeg-turbo/archive/refs/heads/2.0.6.2.zip -O libjpeg-turbo-2.0.6.2.zip
unzip libjpeg-turbo-2.0.6.2.zip 
cd libjpeg-turbo-2.0.6.2
mkdir build 
cd build 
cmake -DCMAKE_INSTALL_PREFIX=/usr  -DCMAKE_BUILD_TYPE=RELEASE  -DENABLE_STATIC=FALSE  -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib .. 
make -j$(nproc) install 
rm -rf /tmp/libjpeg-turbo

# Get released ninja from source
mkdir -p /tmp/ninja 
cd /tmp/ninja
wget -nv https://codeload.github.com/Kitware/ninja/zip/refs/tags/v1.11.1.g95dee.kitware.jobserver-1 -O ninja.zip 
unzip ninja.zip 
cd ninja-1.11.1.g95dee.kitware.jobserver-1 
./configure.py --bootstrap
cp ninja /usr/local/bin/ 
rm -rf /tmp/ninja

# Install FFmpeg from source
wget -qO- https://www.ffmpeg.org/releases/ffmpeg-4.4.2.tar.gz | tar -xzv -C /usr/local

command -v lbzip2 
ln -sf $(command -v lbzip2) /usr/local/bin/compressor || ln -sf $(command -v bzip2) /usr/local/bin/compressor

# Install Google Benchmark 
mkdir -p /tmp/Gbenchmark 
cd /tmp/Gbenchmark 
wget -qO- https://github.com/google/benchmark/archive/refs/tags/v1.6.1.tar.gz | tar xz 
cmake -Sbenchmark-1.6.1 -Bbuild -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBENCHMARK_ENABLE_TESTING=OFF -DCMAKE_CXX_STANDARD=14 
make -j -C build 
cd /tmp/Gbenchmark/build
make install

# Build boost-1.82.0 from source for RPP
# Installing in a non-standard location since the test packages of hipFFT and rocFFT pick up the version of
# the installed Boost library and declare a package dependency on that specific version of Boost.
# For example, if this was installed in the standard location it would declare a dependency on libboost-dev(el)1.82.0
# which is not available as a package in any distro.
# Once this is fixed, we can remove the Boost package from the requirements list and install this
# in the standard location
# export RPP_DEPS_LOCATION=/usr/local/rpp-deps
mkdir -p /tmp/boost-1.82.0 
cd /tmp/boost-1.82.0 
wget -nv https://sourceforge.net/projects/boost/files/boost/1.82.0/boost_1_82_0.tar.bz2 -O ./boost_1_82_0.tar.bz2 
tar -xf boost_1_82_0.tar.bz2 --use-compress-program="/usr/local/bin/compressor" 
cd boost_1_82_0 
./bootstrap.sh --prefix=${RPP_DEPS_LOCATION} --with-python=python3 
./b2 stage -j$(nproc) threading=multi link=shared cxxflags="-std=c++11" 
./b2 install threading=multi link=shared --with-system --with-filesystem 
./b2 stage -j$(nproc) threading=multi link=static cxxflags="-std=c++11 -fpic" cflags="-fpic" 
./b2 install threading=multi link=static --with-system --with-filesystem 
rm -rf /tmp/boost-1.82.0
