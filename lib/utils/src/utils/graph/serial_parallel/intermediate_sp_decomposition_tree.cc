#include "utils/graph/serial_parallel/intermediate_sp_decomposition_tree.h"
#include "utils/containers/extend.h"

namespace FlexFlow {

struct FlattenAST {
  void add_flattened_child_to_parent(
      IntermediateSpDecompositionTree &parent,
      std::variant<IntermediateSpDecompositionTree, Node> const &child) {
    if (std::holds_alternative<Node>(child)) {
      parent.children.push_back(child);
      return;
    }

    IntermediateSpDecompositionTree child_node =
        std::get<IntermediateSpDecompositionTree>(child);

    if (parent.type == child_node.type) {
      extend(parent.children, child_node.children);
    } else {
      parent.children.push_back(child);
    }
  }

  std::variant<IntermediateSpDecompositionTree, Node>
      operator()(IntermediateSpDecompositionTree const &ast_node) {
    IntermediateSpDecompositionTree result(ast_node.type, {});
    for (std::variant<IntermediateSpDecompositionTree, Node> const &child :
         ast_node.children) {
      std::variant<IntermediateSpDecompositionTree, Node> flattened_child =
          flatten_ast(child);
      add_flattened_child_to_parent(result, flattened_child);
    }
    return result;
  }

  std::variant<IntermediateSpDecompositionTree, Node>
      operator()(Node const &ast_node) {
    return ast_node;
  }
};

std::variant<IntermediateSpDecompositionTree, Node> flatten_ast(
    std::variant<IntermediateSpDecompositionTree, Node> const &ast) {
  return std::visit(FlattenAST{}, ast);
}

} // namespace FlexFlow