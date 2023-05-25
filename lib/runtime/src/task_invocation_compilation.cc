#include "task_invocation_compilation.h"
#include "utils/containers.h"

namespace FlexFlow {


TaskArgumentsFormat create_serializable_format(ConcreteArgsFormat const &concrete_args_format,
                                               FutureArgsFormat const &future_args_format,
                                               optional<IndexArgsFormat> const &index_args_format) {
  TaskArgumentsFormat result;
  for (auto const &kv : concrete_args_format.fmts) {
    result.insert(kv);
  }
  for (auto const &kv : future_args_format.fmts) {
    result.insert(kv);
  }
  assert (!index_args_format.has_value());
  return result;
}

Legion::TaskArgument as_task_argument(ConcreteArgsFormat const &concrete_args_format, 
                                      FutureArgsFormat const &future_args_format, 
                                      optional<TensorArgsFormat> const &tensor_args_format,
                                      optional<IndexArgsFormat> const &index_args_format) {
  TaskArgumentsFormat serializable_format = create_serializable_format(concrete_args_format,
                                                                       future_args_format,
                                                                       tensor_args_format,
                                                                       index_args_format);
  *(concrete_args_format.reserved_bytes_for_fmt) = serializable_format;
  return Legion::TaskArgument(concrete_args_format.sez.get_buffer(),
                              concrete_args_format.sez.get_used_bytes());
}

TaskReturnAccessor execute(TensorlessTaskInvocation const &invocation,
                           TensorArgsFormat const &tensor_args_format,
                           ParallelComputationGraph const &pcg,
                           RuntimeBacking const &backing,
                           EnableProfiling enable_profiling) {
  TaskSignature sig = get_signature(invocation.task_id);
  TensorlessTaskBinding binding = invocation.binding;
  /* TensorArgsFormat tensor_args_format = process_tensor_args(sig, pcg, binding); */
  ConcreteArgsFormat concrete_args_format = process_concrete_args(binding);
  FutureArgsFormat future_args_format = process_future_args(binding);
  TaskInvocationArgsFormat task_invocation_args_format = process_task_invocation_args(binding, enable_profiling, backing);
  assert (get_args_of_type<CheckedTypedFutureMap>(binding).empty()); // currently we don't handle these as I don't think they're used anywhere
  if (binding.invocation_type == InvocationType::STANDARD) {
    assert (get_args_of_type<IndexArgSpec>(binding).empty());
    Legion::TaskArgument task_arg = as_task_argument(concrete_args_format,
                                                     future_args_format,
                                                     tensor_args_format);
    Legion::TaskLauncher launcher(invocation.task_id, task_arg);
    add_tensor_requirements(launcher, tensor_args_format);
    Legion::Future returned_future = backing.execute_task(launcher);
    return TaskReturnAccessor(sig.get_return_type(), returned_future);
  } else if (binding.invocation_type == InvocationType::INDEX) {
    parallel_tensor_guid_t index_space_determiner = binding.domain_spec.value();
    ParallelTensorBacking pt_backing = backing.at(index_space_determiner);
    IndexArgsFormat index_args_format = process_index_args(binding, 
                                                           backing.get_domain(pt_backing.parallel_is));
    Legion::TaskArgument task_arg = as_task_argument(concrete_args_format,
                                                     future_args_format,
                                                     tensor_args_format,
                                                     index_args_format);
    IndexTaskLauncher launcher(invocation.task_id,
                               pt_backing.parallel_is,
                               task_arg,
                               as_argument_map(index_args_format),
                               Predicate::TRUE_PRED,
                               false /*must*/,
                               0 /*mapper_id*/,
                               pt_backing.mapping_id.value()
                               );
    add_tensor_requirements(launcher, tensor_args_format);
    FutureMap returned_future = backing.execute_task(launcher);
    return TaskReturnAccessor(sig.get_return_type(), returned_future);
  }
}


}