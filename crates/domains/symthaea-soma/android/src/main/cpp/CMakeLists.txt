cmake_minimum_required(VERSION 3.22)
project(soma_jni C)

# JNI C glue — legacy compatibility surface + isolated assurance-only checked surface.
add_library(soma_jni SHARED
    soma_jni.c
    soma_assurance_jni.c
)

# Link against pre-built Rust library (cross-compiled per ABI)
add_library(symthaea_soma SHARED IMPORTED)
set_target_properties(symthaea_soma PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libsymthaea_soma.so
)

target_link_libraries(soma_jni symthaea_soma log)
