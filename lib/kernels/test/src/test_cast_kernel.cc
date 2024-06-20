#include "doctest/doctest.h"
#include "kernels/cast_kernels.h"
#include "test_utils.h"
#include <type_traits>

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test cast kernel") {
    ffStream_t stream = create_ff_stream();

    Allocator allocator = get_local_memory_allocator();

    TensorShape input_shape =
        make_float_tensor_shape_from_legion_dims({100, 100});
    TensorShape output_shape =
        make_double_tensor_shape_from_legion_dims({100, 100});

    GenericTensorAccessorR input_accessor =
        read_only_accessor_from_write_accessor(
            create_random_filled_accessor_w(input_shape, allocator));

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);
      Kernels::Cast::forward_kernel(stream,
                                    input_accessor,
                                    output_accessor,
                                    DataType::FLOAT,
                                    DataType::DOUBLE);

      std::vector<double> host_double_data =
          load_data_to_host_from_device<double>(
              read_only_accessor_from_write_accessor(output_accessor));

      CHECK(contains_non_zero(host_double_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(output_shape, allocator));
      GenericTensorAccessorW grad_output_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Cast::backward_kernel(stream,
                                     output_accessor,
                                     grad_output_accessor,
                                     DataType::DOUBLE,
                                     DataType::FLOAT);

      std::vector<float> host_grad_float_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(grad_output_accessor));
      CHECK(contains_non_zero(host_grad_float_data));
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
