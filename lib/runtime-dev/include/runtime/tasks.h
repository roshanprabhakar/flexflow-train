/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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
#ifndef _RUNTIME_TASK_H_
#define _RUNTIME_TASK_H_
#include "legion.h"
#ifdef FF_USE_NCCL
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include <nccl.h>
#else
#include <rccl/rccl.h>
#endif
#endif

namespace FlexFlow {

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    CUDA_INIT_TASK_ID,
    NCCL_GETUNIQUEID_TASK_ID,
    NCCL_INIT_COMMS_TASK_ID,
    NCCL_FINISH_COMMS_TASK_ID,
    EXEC_FORWARD_TASK_ID,
    EXEC_BACKWARD_TASK_ID,
    EXEC_OPTIMIZE_TASK_ID,
};

void top_level_task(Legion::Task const *task,
                    std::vector<Legion::PhysicalRegion> const &regions,
                    Legion::Context ctx,
                    Legion::Runtime *runtime);

void register_flexflow_internal_tasks(Legion::Runtime *runtime = NULL,
                                      bool pre_register = true,
                                      bool enable_control_replication = true);

void register_custom_tasks();

} // namespace FlexFlow

#endif // _RUNTIME_TASK_H_
