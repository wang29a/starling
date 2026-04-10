//
// Created by Songlin Wu on 2022/6/30.
// Modified for custom 3-file separated graph format.
//
#include <chrono>
#include <string>
#include <utils.h>
#include <memory>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include <cstring>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <utility>
#include <omp.h>
#include <cmath>
#include <mutex>
#include <queue>
#include <random>

#include "cached_io.h"
#include "pq_flash_index.h"
#include "aux_utils.h"

#ifndef SECTOR_LEN
#define SECTOR_LEN (size_t)4096
#endif

#define READ_SECTOR_OFFSET(node_id) \
  ((_u64) node_id / nnodes_per_sector  + 1) * SECTOR_LEN + ((_u64) node_id % nnodes_per_sector) * max_node_len;
#define INF 0xffffffff

const std::string partition_index_filename = "_tmp.index";

// Get file size
inline size_t get_file_size_custom(const std::string &fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    size_t end_pos = reader.tellg();
    reader.close();
    return end_pos;
  } else {
    std::cout << "Could not open file: " << fname << std::endl;
    return 0;
  }
}

// Write DiskANN sector data according to graph-partition layout
void relayout(const char* indexname, const char* partition_name) {
  _u64                               C;
  _u64                               _partition_nums;
  _u64                               _nd;
  _u64                               max_node_len;
  std::vector<std::vector<unsigned>> layout;
  std::vector<std::vector<unsigned>> _partition;

  std::ifstream part(partition_name);
  part.read((char*) &C, sizeof(_u64));
  part.read((char*) &_partition_nums, sizeof(_u64));
  part.read((char*) &_nd, sizeof(_u64));
  std::cout << "C: " << C << " partition_nums:" << _partition_nums
            << " _nd:" << _nd << std::endl;

  auto meta_pair = diskann::get_disk_index_meta(indexname);
  _u64 actual_index_size = get_file_size_custom(indexname);
  _u64 expected_file_size, expected_npts;

  if (meta_pair.first) {
      expected_file_size = meta_pair.second.back();
      expected_npts = meta_pair.second.front();
  } else {
      expected_file_size = meta_pair.second.front();
      expected_npts = meta_pair.second[1];
  }

  if (expected_file_size != actual_index_size) {
    diskann::cout << "File size mismatch for " << indexname
                  << " (size: " << actual_index_size << ")"
                  << " with meta-data size: " << expected_file_size << std::endl;
    exit(-1);
  }
  if (expected_npts != _nd) {
    diskann::cout << "expect _nd: " << _nd
                  << " actual _nd: " << expected_npts << std::endl;
    exit(-1);
  }
  max_node_len = meta_pair.second[3];
  unsigned nnodes_per_sector = meta_pair.second[4];
  if (SECTOR_LEN / max_node_len != C) {
    diskann::cout << "nnodes per sector: " << SECTOR_LEN / max_node_len << " C: " << C
                  << std::endl;
    exit(-1);
  }

  layout.resize(_partition_nums);
  for (unsigned i = 0; i < _partition_nums; i++) {
    unsigned s;
    part.read((char*) &s, sizeof(unsigned));
    layout[i].resize(s);
    part.read((char*) layout[i].data(), sizeof(unsigned) * s);
  }
  part.close();

  _u64            read_blk_size = 64 * 1024 * 1024;
  _u64            write_blk_size = read_blk_size;

  std::string partition_path(partition_name);
  partition_path = partition_path.substr(0, partition_path.find_last_of('.')) + partition_index_filename;
  cached_ofstream diskann_writer(partition_path, write_blk_size);

  std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
  std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);

  std::cout << "nnodes per sector "<<nnodes_per_sector << std::endl;
  _u64 file_size = SECTOR_LEN + SECTOR_LEN * ((_nd + nnodes_per_sector - 1) / nnodes_per_sector);
  std::unique_ptr<char[]> mem_index =
      std::make_unique<char[]>(file_size);
  std::ifstream diskann_reader(indexname);
  diskann_reader.read(mem_index.get(),file_size);
  std::cout << "C: " << C << " partition_nums:" << _partition_nums
            << " _nd:" << _nd << std::endl;

  const _u64 disk_file_size = _partition_nums * SECTOR_LEN + SECTOR_LEN;
  if (meta_pair.first) {
    char* meta_buf = mem_index.get() + 2 * sizeof(int);
    *(reinterpret_cast<_u64*>(meta_buf + 4 * sizeof(_u64))) = C;
    *(reinterpret_cast<_u64*>(meta_buf + (meta_pair.second.size()-1) * sizeof(_u64)))
        = disk_file_size;
  } else {
    _u64* meta_buf = reinterpret_cast<_u64*>(mem_index.get());
    *meta_buf = disk_file_size;
    *(meta_buf + 4) = C;
  }
  std::cout << "size "<< disk_file_size << std::endl;
  diskann_writer.write((char*) mem_index.get(),
                       SECTOR_LEN);
  for (unsigned i = 0; i < _partition_nums; i++) {
    if (i % 100000 == 0) {
      diskann::cout << "relayout has done " << (float) i / _partition_nums
                    << std::endl;
      diskann::cout.flush();
    }
    memset(sector_buf.get(), 0, SECTOR_LEN);
    for (unsigned j = 0; j < layout[i].size(); j++) {
      unsigned id = layout[i][j];
      memset(node_buf.get(), 0, max_node_len);
      uint64_t index_offset = READ_SECTOR_OFFSET(id);
      uint64_t buf_offset = (uint64_t)j * max_node_len;
      memcpy((char*) sector_buf.get() + buf_offset,
             (char*) mem_index.get() + index_offset, max_node_len);
    }
    diskann_writer.write(sector_buf.get(), SECTOR_LEN);
  }
  diskann::cout << "Relayout index." << std::endl;
}

/**
 * Relayout custom 3-file separated graph topology according to partition layout.
 */
void relayout_custom(const char* graph_file, const char* meta_file,
                     const char* partition_name, const char* output_graph_file,
                     unsigned ep_size = 1) {
  // 1. Read partition file
  _u64 C;
  _u64 _partition_nums;
  _u64 _nd;

  std::ifstream part(partition_name, std::ios::binary);
  if (!part.is_open()) {
    std::cout << "Cannot open partition file: " << partition_name << std::endl;
    exit(-1);
  }
  part.read((char*)&C, sizeof(_u64));
  part.read((char*)&_partition_nums, sizeof(_u64));
  part.read((char*)&_nd, sizeof(_u64));
  std::cout << "Partition info:" << std::endl;
  std::cout << "  C: " << C << std::endl;
  std::cout << "  partition_nums: " << _partition_nums << std::endl;
  std::cout << "  _nd: " << _nd << std::endl;

  // Read layout
  std::vector<std::vector<unsigned>> layout(_partition_nums);
  for (unsigned i = 0; i < _partition_nums; i++) {
    unsigned s;
    part.read((char*)&s, sizeof(unsigned));
    layout[i].resize(s);
    part.read((char*)layout[i].data(), sizeof(unsigned) * s);
  }
  part.close();

  // 2. Read meta file
  std::ifstream meta_in(meta_file, std::ios::binary);
  if (!meta_in.is_open()) {
    std::cout << "Cannot open meta file: " << meta_file << std::endl;
    exit(-1);
  }

  uint32_t node_num, emb_dim, loc_dim, max_nbr_len, max_alpha_range_len;
  uint64_t nnodes_per_sector_meta;

  meta_in.read((char*)&node_num, sizeof(uint32_t));
  meta_in.read((char*)&emb_dim, sizeof(uint32_t));
  meta_in.read((char*)&loc_dim, sizeof(uint32_t));
  meta_in.read((char*)&max_nbr_len, sizeof(uint32_t));
  meta_in.read((char*)&max_alpha_range_len, sizeof(uint32_t));
  meta_in.read((char*)&nnodes_per_sector_meta, sizeof(uint64_t));

  std::vector<uint32_t> enterpoint_set(ep_size);
  meta_in.read((char*)enterpoint_set.data(), ep_size * sizeof(uint32_t));
  meta_in.close();

  std::cout << "Meta info:" << std::endl;
  std::cout << "  node_num: " << node_num << std::endl;
  std::cout << "  max_nbr_len: " << max_nbr_len << std::endl;
  std::cout << "  max_alpha_range_len: " << max_alpha_range_len << std::endl;
  std::cout << "  nnodes_per_sector: " << nnodes_per_sector_meta << std::endl;

  if (node_num != _nd) {
    std::cout << "Node count mismatch!" << std::endl;
    exit(-1);
  }

  // Calculate fixed topology node size
  uint64_t alpha_pairs_size = max_alpha_range_len * 2;
  uint64_t neighbor_size = 4 + alpha_pairs_size;
  uint64_t fixed_topo_size = 4 + max_nbr_len * neighbor_size;

  std::cout << "  fixed_topo_size: " << fixed_topo_size << " bytes" << std::endl;

  // Calculate page layout
  uint64_t page_size = 8192;  // Standard page size
  uint64_t nodes_per_page = page_size / fixed_topo_size;
  uint64_t padding_per_page = page_size - nodes_per_page * fixed_topo_size;

  std::cout << "  page_size: " << page_size << " bytes" << std::endl;
  std::cout << "  nodes_per_page: " << nodes_per_page << std::endl;
  std::cout << "  padding_per_page: " << padding_per_page << " bytes" << std::endl;

  // Verify graph file size with page padding
  size_t actual_graph_size = get_file_size_custom(graph_file);
  uint64_t full_pages = _nd / nodes_per_page;
  uint64_t remainder = _nd % nodes_per_page;
  size_t expected_graph_size = full_pages * page_size + (remainder > 0 ? page_size : 0);

  if (actual_graph_size != expected_graph_size) {
    std::cout << "Graph file size mismatch!" << std::endl;
    std::cout << "  Expected: " << expected_graph_size << std::endl;
    std::cout << "  Actual: " << actual_graph_size << std::endl;
    exit(-1);
  }

  // Helper function to calculate file offset for a node considering page padding
  auto get_node_offset = [&](uint32_t node_id) -> uint64_t {
    uint64_t page_index = node_id / nodes_per_page;
    uint64_t offset_in_page = (node_id % nodes_per_page) * fixed_topo_size;
    return page_index * page_size + offset_in_page;
  };

  // 3. Read entire graph file
  std::unique_ptr<char[]> graph_data = std::make_unique<char[]>(actual_graph_size);
  std::ifstream graph_in(graph_file, std::ios::binary);
  graph_in.read(graph_data.get(), actual_graph_size);
  graph_in.close();

  // 4. Write relayouted graph
  std::ofstream graph_out(output_graph_file, std::ios::binary);
  if (!graph_out.is_open()) {
    std::cout << "Cannot create output graph file: " << output_graph_file << std::endl;
    exit(-1);
  }

  std::cout << "Writing relayouted graph (page by page)..." << std::endl;

  // Allocate page buffer
  std::unique_ptr<char[]> page_buf = std::make_unique<char[]>(page_size);

  for (unsigned i = 0; i < _partition_nums; i++) {
    if (i % 100000 == 0) {
      std::cout << "  Progress: " << (float)i / _partition_nums * 100 << "%" << std::endl;
    }

    // Fill page buffer with this partition's nodes
    memset(page_buf.get(), 0, page_size);  // Clear buffer (for padding)
    for (unsigned j = 0; j < layout[i].size(); j++) {
      unsigned node_id = layout[i][j];
      uint64_t offset = get_node_offset(node_id);
      // Copy node to position j in the page buffer
      memcpy(page_buf.get() + j * fixed_topo_size,
             graph_data.get() + offset, fixed_topo_size);
    }
    // Write full page (with padding if partition not full)
    graph_out.write(page_buf.get(), page_size);
  }
  graph_out.close();

  std::cout << "Relayout complete: " << output_graph_file << std::endl;

  // 5. Copy and update meta file
  std::string output_meta_file = std::string(output_graph_file) + "_meta.bin";
  std::ifstream meta_in2(meta_file, std::ios::binary);
  std::ofstream meta_out(output_meta_file, std::ios::binary);

  char meta_buf[4096];
  meta_in2.read(meta_buf, 4096);

  uint64_t* nnodes_ptr = (uint64_t*)(meta_buf + 5 * sizeof(uint32_t));
  *nnodes_ptr = C;  // Update nnodes_per_sector to new partition size

  meta_out.write(meta_buf, 4096);
  meta_in2.close();
  meta_out.close();

  std::cout << "Updated meta file: " << output_meta_file << std::endl;
}

void print_usage(const char* prog_name) {
  std::cout << "Usage:" << std::endl;
  std::cout << "  DiskANN format:" << std::endl;
  std::cout << "    " << prog_name << " <index_file> <partition_file>" << std::endl;
  std::cout << std::endl;
  std::cout << "  Custom format:" << std::endl;
  std::cout << "    " << prog_name << " --custom <graph_file> <meta_file> <partition_file> <output_graph>" << std::endl;
  std::cout << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --custom  : use custom 3-file separated graph format" << std::endl;
  std::cout << "  --help    : print this help message" << std::endl;
}

int main(int argc, char** argv){
  if (argc < 2) {
    print_usage(argv[0]);
    return -1;
  }

  // Check for --help
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
  }

  // Check for --custom flag
  if (argc >= 2 && std::string(argv[1]) == "--custom") {
    if (argc != 6) {
      std::cerr << "Error: --custom requires 4 arguments" << std::endl;
      print_usage(argv[0]);
      return -1;
    }
    const char* graph_file = argv[2];
    const char* meta_file = argv[3];
    const char* partition_file = argv[4];
    const char* output_graph = argv[5];
    relayout_custom(graph_file, meta_file, partition_file, output_graph);
  } else {
    // Original DiskANN format
    if (argc != 3) {
      std::cerr << "Error: DiskANN format requires 2 arguments" << std::endl;
      print_usage(argv[0]);
      return -1;
    }
    relayout(argv[1], argv[2]);
  }

  return 0;
}
