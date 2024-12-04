#!/bin/bash
set -ex
source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"
set_component_src rocDecode
BUILD_DEV=ON
build_rocdecode() {
    if [ "$DISTRO_ID" = "centos-7" ] || [ "$DISTRO_ID" = "sles-15.4" ] ; then
     echo "Not building rocDecode for ${DISTRO_ID}. Exiting..." 
     return 0
    fi

    echo "Start build"
    mkdir -p $BUILD_DIR && cd $BUILD_DIR
    python3 ${COMPONENT_SRC}/rocDecode-setup.py --developer OFF
    
    cmake -DROCM_DEP_ROCMCORE=ON ${COMPONENT_SRC}
    make -j8
    make install
    make package
    
    cmake --build "$BUILD_DIR" -- -j${PROC}
    cpack -G ${PKGTYPE^^} -B ${BUILD_DIR}
    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR
    cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR
    show_build_cache_stats
}
clean_rocdecode() {
    echo "Cleaning rocDecode build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}
stage2_command_args "$@"
case $TARGET in
    build) build_rocdecode ;;
    outdir) print_output_directory ;;
    clean) clean_rocdecode ;;
    *) die "Invalid target $TARGET" ;;
esac
