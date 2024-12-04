#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src hipSOLVER

build_hipsolver() {
    echo "Start build"

    cd $COMPONENT_SRC

    CXX="g++"
    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
       set_asan_env_vars
       set_address_sanitizer_on
    fi

    echo "C compiler: $CC"
    echo "CXX compiler: $CXX"
    echo "FC compiler: $FC"

    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
       rebuild_lapack
    fi

    init_rocm_common_cmake_params
    cmake \
        -DUSE_CUDA=OFF \
        ${LAUNCHER_FLAGS} \
        "${rocm_math_common_cmake_params[@]}" \
        -DBUILD_CLIENTS_TESTS=ON \
        -DBUILD_CLIENTS_BENCHMARKS=ON \
        -DBUILD_CLIENTS_SAMPLES=ON \
        -DCPACK_SET_DESTDIR=OFF \
        -DBUILD_ADDRESS_SANITIZER="${ADDRESS_SANITIZER}" \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cmake --build "$BUILD_DIR" -- package

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR

    show_build_cache_stats
}

clean_hipsolver() {
    echo "Cleaning hipSOLVER build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_hipsolver ;;
    outdir) print_output_directory ;;
    clean) clean_hipsolver ;;
    *) die "Invalid target $TARGET" ;;
esac
