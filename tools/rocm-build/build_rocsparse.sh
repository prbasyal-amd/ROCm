

set -ex
source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

PATH=${ROCM_PATH}/bin:$PATH
set_component_src rocSPARSE

build_rocsparse() {
    echo "Start build"

    cd $COMPONENT_SRC

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
       set_asan_env_vars
       set_address_sanitizer_on
    fi

    MIRROR="http://compute-artifactory.amd.com/artifactory/list/rocm-generic-local/mathlib/sparse/"

    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

    if [ -n "$GPU_ARCHS" ]; then
        GPU_TARGETS="$GPU_ARCHS"
    else
        GPU_TARGETS="gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101"
    fi

    ROCSPARSE_TEST_MIRROR=$MIRROR \
    export CXX=$(set_build_variables CXX)\
    export CC=$(set_build_variables CC)\

    init_rocm_common_cmake_params
    cmake \
        -DAMDGPU_TARGETS=${GPU_TARGETS} \
        ${LAUNCHER_FLAGS} \
        "${rocm_math_common_cmake_params[@]}"\
        -DBUILD_CLIENTS_SAMPLES=ON \
        -DBUILD_CLIENTS_TESTS=ON \
        -DBUILD_CLIENTS_BENCHMARKS=ON \
        -DCPACK_SET_DESTDIR=OFF \
        -DCMAKE_INSTALL_PREFIX=${ROCM_PATH} \
        -DBUILD_ADDRESS_SANITIZER="${ADDRESS_SANITIZER}" \
        -DCMAKE_MODULE_PATH="${ROCM_PATH}/lib/cmake/hip;${ROCM_PATH}/hip/cmake" \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cmake --build "$BUILD_DIR" -- package

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR

    show_build_cache_stats
}

clean_rocsparse() {
    echo "Cleaning rocSPARSE build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_rocsparse ;;
    outdir) print_output_directory ;;
    clean) clean_rocsparse ;;
    *) die "Invalid target $TARGET" ;;
esac
