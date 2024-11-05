/* Copyright 2023 CMU, Stanford, Facebook, LANL
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

#include "flexflow/page_manager.h"

namespace FlexFlow {

// For all runtime functions, they share a single page manager for pages
// information
PageManager *page_manager_singleton = nullptr;

// the interface of logicaltokenblock
LogicalTokenBlock::LogicalTokenBlock(int block_number, uint32_t block_size)
    : block_number(block_number), block_size(block_size), num_tokens(0),
      num_commit_tokens(0), num_spec_tokens(0) {}

bool LogicalTokenBlock::is_empty() const {
  assert(num_spec_tokens == 0 && num_commit_tokens == 0);
  assert(num_tokens <= block_size);
  return num_tokens == 0;
}

bool LogicalTokenBlock::is_full() const {
  assert(num_spec_tokens + num_commit_tokens == num_tokens);
  assert(num_tokens <= block_size);
  return num_tokens == block_size;
}

int LogicalTokenBlock::get_num_empty_slots() const {
  assert(num_spec_tokens + num_commit_tokens == num_tokens);
  assert(num_tokens <= block_size);
  return block_size - num_tokens;
}

int LogicalTokenBlock::get_num_alloc_slots() const {
  assert(num_spec_tokens + num_commit_tokens == num_tokens);
  assert(num_tokens <= block_size);
  return num_tokens;
}

void LogicalTokenBlock::reset_num_spec_tokens() {
  assert(num_spec_tokens + num_commit_tokens == num_tokens);
  assert(num_tokens <= block_size);

  num_tokens -= num_spec_tokens;
  num_spec_tokens = 0;

  assert(num_spec_tokens + num_commit_tokens == num_tokens);
  assert(num_tokens <= block_size);
}

void LogicalTokenBlock::append_tokens(
    std::vector<TokenId> const &token_ids_to_append, bool committed) {
  assert(num_spec_tokens + num_commit_tokens == num_tokens);
  assert(num_tokens <= block_size);
  if (num_tokens + token_ids_to_append.size() > block_size) {
    printf("block is full! Cannot append more tokens\n");
    throw std::runtime_error("Block is full! Cannot append more tokens.");
  }
  token_ids.insert(
      token_ids.end(), token_ids_to_append.begin(), token_ids_to_append.end());
  num_tokens += token_ids_to_append.size();
  if (committed) {
    num_commit_tokens += token_ids_to_append.size();
  } else {
    num_spec_tokens += token_ids_to_append.size();
  }
  assert(num_spec_tokens + num_commit_tokens == num_tokens);
  assert(num_tokens <= block_size);
}

std::vector<TokenId> LogicalTokenBlock::get_token_ids() const {
  return token_ids;
}

PhysicalTokenBlock::PhysicalTokenBlock(int block_number, int block_size)
    : block_number(block_number), block_size(block_size), ref_count(0) {}

BlockAllocator::BlockAllocator(int block_size, int num_total_blocks) {
  for (int block_number = 0; block_number < num_total_blocks; ++block_number) {
    free_blocks.push_back(PhysicalTokenBlock(block_number, block_size));
  }
  num_total_blocks = num_total_blocks;
}

// Allocate a block
PhysicalTokenBlock BlockAllocator::allocate() {
  if (free_blocks.empty()) {
    printf("no free blocks are available\n");
    throw std::runtime_error("Out of memory! No free blocks are available.");
  }
  PhysicalTokenBlock block = free_blocks.front();
  free_blocks.pop_front();
  block.incr_ref_count();
  return block;
}

// Free a block
void BlockAllocator::free(PhysicalTokenBlock &block) {
  if (block.ref_count == 0) {
    printf("block is already freed\n");
    throw std::runtime_error("Double free! Block is already freed.");
  }
  block.decr_ref_count();
  if (block.ref_count == 0) {
    // printf("put block number: %d back to free_blocks\n",
    // block.get_block_number());
    free_blocks.push_back(block);
  } else {
    // in current implementation this should not be the case
    printf("block is not freed. Ref count: %d\n", block.ref_count);
    throw std::runtime_error("Block is not freed. Ref count: " +
                             std::to_string(block.ref_count));
  }
}

int BlockAllocator::get_num_free_blocks() const {
  return free_blocks.size();
}

PageManager::PageManager(int block_size, int num_total_blocks)
    : block_size(block_size), num_total_blocks(num_total_blocks),
      block_allocator(block_size, num_total_blocks) {}

// return the physical number of this block
int PageManager::allocate_one_block(RequestGuid const &request_guid) {
  BlockTable &block_table = block_tables[request_guid];

  PhysicalTokenBlock block = block_allocator.allocate();
  block_table.push_back(block);
  block_tables[request_guid] = block_table;
  return block.get_block_number();
}

void PageManager::free_block_table(BlockTable &block_table) {
  // make it reverse order to free the last allocated block first
  BlockTable::reverse_iterator rit = block_table.rbegin();
  for (; rit != block_table.rend(); ++rit) {
    block_allocator.free(*rit);
  }
  return;
}

void PageManager::free_request(RequestGuid const &request_guid) {
  // we only free the blocks that are already used
  assert(block_tables.find(request_guid) != block_tables.end());
  BlockTable block_table = block_tables[request_guid];
  free_block_table(block_table);
  block_tables.erase(request_guid);
  return;
}

// delete the last num_blocks in the request_guid
void PageManager::free_multiple_blocks(RequestGuid const &request_guid,
                                       int num_blocks) {
  assert(block_tables.find(request_guid) != block_tables.end());
  auto &block_table = block_tables[request_guid];
  assert(num_blocks <= block_table.size());
  int num_blocks_allocated = block_table.size();
  for (int i = 0; i < num_blocks; i++) {
    block_allocator.free(block_table[num_blocks_allocated - i - 1]);
  }
  // only keep the first num_blocks_allocated - num_blocks blocks
  block_table.erase(block_table.begin() + num_blocks_allocated - num_blocks,
                    block_table.end());
  block_tables[request_guid] = block_table;
  return;
}

// int PageManager::get_index_last_block(const RequestGuid& request_guid) const
// {
//     const auto& block_table = block_tables.at(request_guid);
//     return block_table.back.get_block_number();
// }

std::vector<int> PageManager::get_block_table_indices(
    RequestGuid const &request_guid) const {
  std::vector<int> indices;
  auto const &it = block_tables.find(request_guid);
  if (it == block_tables.end()) {
    return indices;
  }
  auto const &block_table = it->second;
  for (auto const &block : block_table) {
    indices.push_back(block.get_block_number());
  }
  return indices;
}

int PageManager::get_num_total_free_blocks() const {
  return block_allocator.get_num_free_blocks();
}

int PageManager::get_num_allocated_blocks(
    RequestGuid const &request_guid) const {
  auto it = block_tables.find(request_guid);
  if (it == block_tables.end()) {
    return 0;
  } else {
    return it->second.size();
  }
}

PageManager *PageManager::get_page_manager(FFModel *ff,
                                           int total_kv_cache_size) {
  int num_kv_heads = ff->num_kv_heads;
  int size_dt = ff->size_dt;
  int qkv_dim = ff->qkv_dim;
  int num_transformer_layers = ff->num_transformer_layers;
  int pipeline_parallelism_degree = ff->config.pipeline_parallelism_degree;
  printf("num_kv_heads: %d, size_dt: %d, qkv_dim: %d, num_transformer_layers: "
         "%d, pipeline_parallelism_degree: %d\n",
         num_kv_heads, size_dt, qkv_dim, num_transformer_layers,
         pipeline_parallelism_degree);
  assert(num_kv_heads > 0 && size_dt > 0 && qkv_dim > 0 &&
         num_transformer_layers > 0 && pipeline_parallelism_degree > 0); //needs to make sure that the model is initialized
  printf("page manager singleton is initialized\n");
  if (page_manager_singleton == nullptr) {
    int num_total_blocks = 0;
    if (total_kv_cache_size == -1) {
      num_total_blocks = (BatchConfig::max_spec_tree_token_num() +
                          BatchConfig::max_sequence_length() + kPagesize - 1) /
                         kPagesize * BatchConfig::max_requests_per_batch();
    } else {
      num_total_blocks =
          total_kv_cache_size * 1024 * 1024 / size_dt / qkv_dim / num_kv_heads /
          (num_transformer_layers / pipeline_parallelism_degree) / 2;
    }
    page_manager_singleton = new PageManager(kPagesize, num_total_blocks);
  }
  return page_manager_singleton;
}

PageManager *PageManager::get_page_manager() {
  // if (page_manager_singleton == nullptr) {
  //   int num_total_blocks =
  //       (BatchConfig::max_spec_tree_token_num() +
  //        BatchConfig::max_sequence_length() + kPagesize - 1) /
  //       kPagesize * BatchConfig::max_requests_per_batch();
  //   page_manager_singleton = new PageManager(kPagesize, num_total_blocks);
  // }
  assert(page_manager_singleton != nullptr);
  return page_manager_singleton;
}

}; // namespace FlexFlow