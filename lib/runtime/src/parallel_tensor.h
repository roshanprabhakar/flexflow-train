/**
 * @file parallel_tensor.h
 * @brief Parallel Tensor Representation
 *
 * @copyright Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford
 * (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_H
#define _FLEXFLOW_RUNTIME_SRC_PARALLEL_TENSOR_H

#include "op-attrs/ffconst.h"
#include "pcg/machine_view.h"
#include "utils/record_formatter.h"
#include <ostream>
#include <unordered_map>
#include "utils/strong_typedef.h"
#include "create_grad.h"
#include "initializer.h"
#include "parallel_tensor_guid_t.h"

namespace FlexFlow {

class FFConfig;

/**
 * @brief Base structure of the parallel tensor representation.
 *
 * @details Parallel tensor is the fundamental component to support the
 * representation and exploration of parallelization strategies.
 */
struct ParallelTensor {
  ParallelTensor() = delete;
  ParallelTensor(ParallelTensor const &rhs);

  ParallelTensor(ParallelTensorShape const &,
                 CreateGrad create_gradients,
                 optional<ParameterSyncType> sync_type = nullopt,
                 Initializer *initializer = nullptr);

  /* void inline_map(FFConfig &config); */
  /* void inline_unmap(FFConfig &config); */
  /* template <typename T> */
  /* T *get_raw_ptr(FFConfig &config); */
  /* void attach_raw_ptr(FFConfig &config, void *raw_ptr, bool column_major); */
  /* void detach_raw_ptr(FFConfig &config); */
  bool get_input_sub_tensor(MachineView const &,
                            ParallelTensor &tensor,
                            OperatorType type);
  bool get_sub_tensor(MachineView const &mv,
                      ParallelTensor &subtensor) const;
  bool get_output_sub_tensor(MachineView const &,
                             ParallelTensor &tensor,
                             OperatorType type);
  size_t get_owner_independent_hash() const;
  size_t get_volume() const;
  size_t get_total_num_parts() const;
  int get_num_replica_dims() const;
  int get_num_replicas() const;
  /* Legion::Domain get_domain() const; */
  bool check_valid() const;
  bool is_valid_machine_view(MachineView const &view) const;
  void print(std::string const &name) const;
  static bool update_parallel_ids(int numdim, ParallelDim *dims);
  ParallelTensorShape get_shape() const;

private:
  template <typename T>
  bool get_input_sub_tensor_via_mappings(MachineView const &,
                                         ParallelTensor &tensor) const;

public:
  ParallelTensorDims dims;
  DataType data_type;
  ParameterSyncType sync_type = ParameterSyncType::NONE;
  optional<Initializer> initializer = nullopt;
  bool create_gradients = false;
};
using ParallelParameter = ParallelTensor;

}

VISITABLE_STRUCT(::FlexFlow::ParallelTensor, dims, data_type, sync_type, initializer, create_gradients);

namespace FlexFlow {
static_assert(std::is_copy_constructible<ParallelTensor>::value, "");
}

#endif