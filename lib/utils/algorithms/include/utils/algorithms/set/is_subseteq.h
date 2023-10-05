#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_SET_IS_SUBSETEQ_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_SET_IS_SUBSETEQ_H

#include <unordered_set>

namespace FlexFlow {

template <typename T>
bool is_subseteq_of(std::unordered_set<T> const &l,
                    std::unordered_set<T> const &r) {
  if (l.size() > r.size()) {
    return false;
  }

  for (auto const &ll : l) {
    if (!contains(r, ll)) {
      return false;
    }
  }
  return true;
}

} // namespace FlexFlow

#endif