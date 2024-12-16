ff_add_library(
  NAME
    runtime-dev
  SRC_PATTERNS
    src/*.cc
  PUBLIC_INCLUDE
    include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../deps/legion/runtime

  PRIVATE_INCLUDE
    src/
  DEPS
    op-attrs
    utils
    legion
    compiler
    kernels
    pcg
    local-execution
)

# ff_add_test_executable(
#   NAME
#     runtime-test
#   SRC_PATTERNS
#     test/src/*.cc
#   PUBLIC_INCLUDE
#     include/
#   PRIVATE_INCLUDE
#     test/src/ src/
#   DEPS
#     runtime
#     doctest
# )
