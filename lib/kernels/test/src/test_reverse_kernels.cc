#include "doctest/doctest.h"
#include "kernels/reverse_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reverse Forward and Backward Kernels") {
    std::size_t reverse_dim_size = 10;
    std::size_t in_blk_size = 10;
    std::size_t num_out_blks = 1;

    TensorShape shape = make_float_tensor_shape_from_legion_dims({100});

    ffStream_t stream = create_ff_stream();

    Allocator allocator = get_local_memory_allocator();

    GenericTensorAccessorW grad_input_accessor =
        create_filled_accessor_w(shape, allocator, 0.0f);
    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w(shape, allocator, 1.0f));
      GenericTensorAccessorW output_accessor = allocator.allocate_tensor(shape);

      Kernels::Reverse::forward_kernel(stream,
                                       input_accessor.get_float_ptr(),
                                       output_accessor.get_float_ptr(),
                                       num_out_blks,
                                       reverse_dim_size,
                                       in_blk_size,
                                       input_accessor.shape.num_elements());

      std::vector<float> check_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      CHECK(contains_non_zero(check_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW output_accessor =
          create_random_filled_accessor_w(shape, allocator);
      Kernels::Reverse::backward_kernel(stream,
                                        output_accessor.get_float_ptr(),
                                        grad_input_accessor.get_float_ptr(),
                                        num_out_blks,
                                        reverse_dim_size,
                                        in_blk_size,
                                        output_accessor.shape.num_elements());

      std::vector<float> host_grad_input_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(grad_input_accessor));
      CHECK(contains_non_zero(host_grad_input_data));
    }

    checkCUDA(cudaStreamDestroy(stream));
  }
}
