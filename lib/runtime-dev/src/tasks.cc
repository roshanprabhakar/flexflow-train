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

#include "runtime/tasks.h"

using namespace Legion;

namespace FlexFlow {

void register_flexflow_internal_tasks(Runtime *runtime,
                                      bool pre_register,
                                      bool enable_control_replication) {
  if (!pre_register) {
    assert(runtime != NULL);
  }
  // CUDA_INIT_TASK
  {
    TaskVariantRegistrar registrar(CUDA_INIT_TASK_ID, "cuda_init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    if (pre_register) {
      Runtime::preregister_task_variant<FFHandler,
                                        UtilityTasks::init_cuda_task>(
          registrar, "cuda_init_task");
    } else {
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      runtime->register_task_variant<FFHandler, UtilityTasks::init_cuda_task>(
          registrar);
    }
  }
#ifdef FF_USE_NCCL
  // NCCL
  {
    TaskVariantRegistrar registrar(NCCL_GETUNIQUEID_TASK_ID,
                                   "NCCL GetUniqueId");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    if (pre_register) {
      Runtime::preregister_task_variant<ncclUniqueId,
                                        Op::get_nccl_unique_id_task>(
          registrar, "NCCL GetUniqueId Task");
    } else {
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      runtime->register_task_variant<ncclUniqueId, Op::get_nccl_unique_id_task>(
          registrar);
    }
  }
  {
    TaskVariantRegistrar registrar(NCCL_INIT_COMMS_TASK_ID,
                                   "NCCL Init Communicators");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    registrar.set_concurrent();
    // registrar.set_concurrent_barrier();
    if (pre_register) {
      Runtime::preregister_task_variant<ncclComm_t, Op::init_nccl_comms_task>(
          registrar, "NCCL Init Communicators Task", 111 /*variant ID*/);
    } else {
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      runtime->register_task_variant<ncclComm_t, Op::init_nccl_comms_task>(
          registrar, 111 /*variant ID*/);
    }
  }
  {
    TaskVariantRegistrar registrar(NCCL_FINISH_COMMS_TASK_ID,
                                   "NCCL Finish Communicators");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    registrar.set_concurrent();
    // registrar.set_concurrent_barrier();
    if (pre_register) {
      Runtime::preregister_task_variant<Op::finish_nccl_comms_task>(
          registrar, "NCCL Finish Communicators Task", 111 /*variant ID*/);
    } else {
      if (enable_control_replication) {
        registrar.global_registration = false;
      }
      runtime->register_task_variant<Op::finish_nccl_comms_task>(
          registrar, 111 /*variant ID*/);
    }
  }
#endif
}

} // namespace FlexFlow
