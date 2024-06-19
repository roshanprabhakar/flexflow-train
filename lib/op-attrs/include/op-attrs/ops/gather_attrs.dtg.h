// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/gather_attrs.struct.toml
/* proj-data
{
  "generated_from": "4ba46b6b494a7a52edda437d2a05fcf1"
}
*/

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_GATHER_ATTRS_DTG_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_GATHER_ATTRS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/ff_dim.dtg.h"
#include "op-attrs/ff_dim.h"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct GatherAttrs {
  GatherAttrs() = delete;
  explicit GatherAttrs(::FlexFlow::ff_dim_t const &dim);

  bool operator==(GatherAttrs const &) const;
  bool operator!=(GatherAttrs const &) const;
  bool operator<(GatherAttrs const &) const;
  bool operator>(GatherAttrs const &) const;
  bool operator<=(GatherAttrs const &) const;
  bool operator>=(GatherAttrs const &) const;
  ::FlexFlow::ff_dim_t dim;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::GatherAttrs> {
  size_t operator()(::FlexFlow::GatherAttrs const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::GatherAttrs> {
  static ::FlexFlow::GatherAttrs from_json(json const &);
  static void to_json(json &, ::FlexFlow::GatherAttrs const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<::FlexFlow::GatherAttrs> {
  static Gen<::FlexFlow::GatherAttrs> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(GatherAttrs const &);
std::ostream &operator<<(std::ostream &, GatherAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_GATHER_ATTRS_DTG_H
