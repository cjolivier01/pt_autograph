#pragma once
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef PYTORCH_PTAG_SCOPE_H_INCLUDED
#define PYTORCH_PTAG_SCOPE_H_INCLUDED

#include <string>
#include <sstream>
#include <unordered_map>
#include <cstring>
#include <algorithm>
#include <vector>
#include <atomic>
#include <limits>
#include <cassert>
#include <stack>

#ifndef PYTORCH_PTAG_NO_FA_SETTER
#include "tensorflow/compiler/xla/client/xla_builder.h"
#endif


#define PYTORCH_PTAG_PUBLIC __attribute__((visibility("default")))

#define PYTORCH_PTAG_THREAD_LOCAL thread_local

namespace pytorch_ptag {

enum class Direction {
  AUTO, FWD, BWD, NONE
};

/**
 * @brief Class to push and pop frontend attributes to a stach which
 *        will be added to graph nodes within the stack's scope.
 *        These attributes will end up in the generated XLA nodes
 *        during XLA lowering.
 */
class FrontendAttributePusher {

 public:
  struct FrontendAttributeContext {
    using seq_nr_t = int64_t;
    std::unordered_map<std::string, std::string> attributes;
    std::atomic<std::size_t> scope_depth{0};
    std::atomic<std::size_t> attribute_id{0};
    std::atomic<std::size_t> in_autograd_thread{0};
    std::stack<seq_nr_t> node_sequence_nr;
    Direction direction{Direction::AUTO};

    struct BackpropFor {
      void *tf_graph = nullptr;
      void *graph = nullptr;
      std::string op_name;
      // PyTorch grad Node sequence number
      seq_nr_t seq_nr{-1};
      void clear() {
        tf_graph = graph = nullptr;
        seq_nr = -1;
        op_name.clear();
      }
    };
    BackpropFor backprop_for_op;
  };

  /**
   * @brief Push frontend attributes to the stack
   * @param values key/value pairs
   * @param id option id number to append to the key
   */
  FrontendAttributePusher(
      const std::unordered_map<std::string, std::string>& values,
      std::size_t id = std::numeric_limits<std::size_t>::max()
  ) {
    InitForceEnable();
    if (IsEnabled()) {
      keys_ = PushFrontendAttributes(
          values,
          id
      );
    }
  }

  /**
   * @brief Push a frontend attribute to the stack
   * @param key attribute name
   * @param value attribute value
   * @param prefix_depth Whether to prefix the key with the current scope depth
   * @param id option id number to append to the key
   */
  FrontendAttributePusher(
      std::string key,
      std::string value,
      std::size_t id = std::numeric_limits<std::size_t>::max()
  ) {
    InitForceEnable();
    if (IsEnabled() > 0) {
      keys_ = {PushFrontendAttribute(std::move(key), std::move(value), id)};
    }
  }

  virtual ~FrontendAttributePusher() {
    PopFrontendAttributes(keys_);
  }

  /**
   * @brief Get the frontend attribute map
   */
  static const std::unordered_map<std::string, std::string> &GetFrontendAttributes() {
    return g_frontend_attribute_context.attributes;
  }

  /**
   * @brief Reset counters and state for the current thread
   */
  static void Reset() {
    if (!g_frontend_attribute_context.attributes.empty()) {
      std::cerr << "[pt_autograph SCOPE WARNING]: Not all frontend attributes were popped"
                << std::endl;
    }
    g_frontend_attribute_context.attribute_id = 0;
    g_frontend_attribute_context.attributes.clear();
    g_frontend_attribute_context.direction = Direction::AUTO;
    g_frontend_attribute_context.backprop_for_op.clear();
  }

  /**
   * @brief Push a key/value pair into the frontend attribute stack
   */
  static std::string PushFrontendAttribute(
      const std::string &key,
      const std::string &value,
      bool change_scope,
      std::size_t id = std::numeric_limits<std::size_t>::max()
  ) {
    if (IsEnabled()) {
      assert(!key.empty());
      std::string emit_key = MakeKeyName(
          key,
          GetCurrentDepth(),
          id == std::numeric_limits<std::size_t>::max()
      );
      assert(!emit_key.empty());
      if (id != std::numeric_limits<std::size_t>::max()) {
        emit_key.append(".");
        // Specified id's become negative
        emit_key.append(std::to_string(-static_cast<long>(id)));
      }
      assert(!g_frontend_attribute_context.attributes.count(emit_key));
      g_frontend_attribute_context.attributes.emplace(emit_key, value);
      if (change_scope) {
        ++g_frontend_attribute_context.scope_depth;
      }
      return emit_key;
    } else {
      return "";
    }
  }

  /**
   * @brief Pop a key/value pair from the frontend attribute stack
   *        The "stack" in this case is conceptual for the "pop", since
   *        the attribute is removed by name, although it may (if specified)
   *        adjust the scope depth, which is analogous to the stack depth
   */
  static void PopFrontendAttribute(
      const std::string &key,
      bool change_scope
  ) {
    if (IsEnabled()) {
      if (!g_frontend_attribute_context.attributes.count(key)) {
        std::cerr << "Attempt to pop unknown frontend attribute key: " << key << std::endl;
        // TF codebase doesn't allow exceptions
        return;
      }
      g_frontend_attribute_context.attributes.erase(key);
      if (change_scope) {
        --g_frontend_attribute_context.scope_depth;
      }
    }
  }

  /**
   * @brief Pop a list of frontend attributes. By using this function,
   *        you can remove multiple key/value pairs while only affecting
   *        the scope depth once (i.e all key/values at same scope depth)
   */
  static void PopFrontendAttributes(const std::vector<std::string> &keys) {
    if (IsEnabled()) {
      for (const std::string &key : keys) {
        PopFrontendAttribute(key, false);
      }
      --g_frontend_attribute_context.scope_depth;
    }
  }

  /**
   * @brief Push a map of frontend attributes. By using this function,
   *        you can add multiple key/value pairs while only affecting
   *        the scope depth once (i.e all key/values at same scope depth)
   */
  static std::vector<std::string> PushFrontendAttributes(
      const std::unordered_map<std::string, std::string>& attribute_map,
      std::size_t id = std::numeric_limits<std::size_t>::max()
  ) {
    if (IsEnabled()) {
      std::vector<std::string> emitted_keys;
      emitted_keys.reserve(attribute_map.size());
      for (const auto &item : attribute_map) {
        emitted_keys.emplace_back(
            PushFrontendAttribute(item.first, item.second, false, id)
        );
      }
      ++g_frontend_attribute_context.scope_depth;
      return std::move(emitted_keys);
    } else {
      return {};
    }
  }

  static void PushNodeSequenceNumber(FrontendAttributeContext::seq_nr_t node_seq_nr) {
    g_frontend_attribute_context.node_sequence_nr.push(node_seq_nr);
  }
  static void PopNodeSequenceNumber() {
    assert(!g_frontend_attribute_context.node_sequence_nr.empty());
    g_frontend_attribute_context.node_sequence_nr.pop();
  }

  static bool GetNodeSequenceNumber(FrontendAttributeContext::seq_nr_t *seq_nr_ptr) {
    if (g_frontend_attribute_context.node_sequence_nr.empty()) {
      return false;
    }
    *seq_nr_ptr = g_frontend_attribute_context.node_sequence_nr.top();
    return true;
  }

  /**
   * @brief Signifies whether we are recording frontend attributes
   */
  static bool IsEnabled() {
    return force_enable_ || enable_count_ > 0;
  }

  /**
   * @brief Get (presumably) scoped enable counter
   */
  static std::size_t GetEnableCount() {
    return enable_count_;
  }

  /**
   * @brief Increment (presumably) scoped enable counter
   */
  static std::size_t IncrementEnabledCount() {
    return ++enable_count_;
  }

  /**
   * @brief Decrement (presumably) scoped enable counter
   */
  static std::size_t DecrementEnabledCount() {
    return --enable_count_;
  }

  /**
   * @brief Returns 'true' if this is the backward pass
   */
  static bool IsAutogradThread() {
    return g_frontend_attribute_context.in_autograd_thread.load() > 0;
  }

  static const FrontendAttributeContext::BackpropFor& GetBackpropOperationFor() {
    return g_frontend_attribute_context.backprop_for_op;
  }

  static void SetBackpropOperationFor(FrontendAttributeContext::BackpropFor op) {
    g_frontend_attribute_context.backprop_for_op = std::move(op);
  }

  /**
   * @brief RAII scope for a backward pass in the current thread
   */
  struct InAutogradThread {
    InAutogradThread() {
      ++g_frontend_attribute_context.in_autograd_thread;
    }
    ~InAutogradThread() {
      --g_frontend_attribute_context.in_autograd_thread;
    }
  };

  /**
   * @brief Returns whether the current statew ius the forward or backward pass
   */
  static Direction GetDirection() {
    return g_frontend_attribute_context.direction;
  }

  /**
   * @brief Directly set the current thread's state to either the
   *        forward or backward pass
   */
  static Direction SetDirection(Direction dir) {
    Direction old_dir = g_frontend_attribute_context.direction;
    g_frontend_attribute_context.direction = dir;
    return old_dir;
  }

  static inline std::string DirectionToString(Direction dir) {
    switch(dir) {
      case Direction::FWD:
        return "FWD";
      case Direction::BWD:
        return "BWD";
      case Direction::AUTO:
        return "AUTO";
      default:
        return "INVALID";
    }
  }

  static inline Direction StringToDirection(const std::string& dir_string) {
    if (dir_string == "FWD") {
      return Direction::FWD;
    }
    if (dir_string == "BWD") {
      return Direction::BWD;
    }
    if (dir_string == "AUTO") {
      return Direction::AUTO;
    }
    assert(false);
    return Direction::FWD;
  }

 private:

  static std::size_t GetCurrentDepth() {
    return g_frontend_attribute_context.scope_depth.load();
  }

  static std::string MakeKeyName(
      const std::string &key,
      std::size_t current_depth,
      bool suffix_id
  ) {
    std::stringstream ss;
    ss << current_depth << '.' << key;
    if (suffix_id) {
      ss << '.' << ++g_frontend_attribute_context.attribute_id;
    }
    return ss.str();
  }

  static void InitForceEnable() {
    const char *s = ::getenv("PYTORCH_PTAG_FORCE_ENABLE_FATTR");
    if (!s) {
      return;
    }
    if (*s == '1') {
      force_enable_ = true;
    } else {
      force_enable_ = false;
    }
  }

  std::vector<std::string> keys_;
  static PYTORCH_PTAG_PUBLIC PYTORCH_PTAG_THREAD_LOCAL FrontendAttributeContext g_frontend_attribute_context;
  static signed int PYTORCH_PTAG_PUBLIC PYTORCH_PTAG_THREAD_LOCAL enable_count_;
  static bool PYTORCH_PTAG_PUBLIC PYTORCH_PTAG_THREAD_LOCAL force_enable_;
};

/**
 * @brief Utility class for generating names for frontend attribute
 *        keys and values
 *        This class need not be used, as key/values can be anything
 *        that the user desires.
 */
class PartitionScope {
 public:
  static constexpr const char *PARTITION_PREFIX = "aten_";
  static constexpr const char *MATCHED_OP = "MATCHED_OP";
  static constexpr const char *REVERSE_OF_OP = "REVERSE_OF_OP";
  static constexpr const char *MATCHED_XLA = "MATCHED_XLA";

  static inline std::string MakeKeyName(
      const char *key_base_name,
      Direction dir = Direction::NONE
  ) {
    if (!FrontendAttributePusher::IsEnabled()) {
      return "";
    }
    std::stringstream ss;
    ss << key_base_name;
    if (dir != Direction::NONE) {
      if (dir == Direction::AUTO) {
        dir = infer_direction();
      }
      ss << "." << FrontendAttributePusher::DirectionToString(dir);
    }
    return ss.str();
  }

  /**
   * @brief Generate a "MATCHED_OP" attribute name
   */
  static inline std::string PartitionMatchName(
      Direction dir,
      const char *match_op_tag = MATCHED_OP
  ) {
    assert(dir != Direction::NONE);
    return MakeKeyName(match_op_tag, dir);
  }

  /**
   * @brief Generate a "MATCHED_OP" attribute's partition name
   */
  static inline std::string MakePartitionName(
      const std::string &function_name,
      const char *prefix = PARTITION_PREFIX
  ) {
    if (!FrontendAttributePusher::IsEnabled()) {
      return "";
    }
    std::stringstream ss;
    ss << prefix << short_fn_name(function_name) /*<< "<float16>(NAT,NAT)"*/;
    return ss.str();
  }

  /**
   * @brief Convert a compiler-macro function name to something "prettier",
   *        such as removing the namespace/class names and parameter portion
   */
  static inline std::string method_name(const std::string& pretty_function) {
    if (FrontendAttributePusher::IsEnabled()) {
      const std::size_t colons = pretty_function.find("::");
      const std::size_t begin = pretty_function.substr(0, colons).rfind(" ") + 1;
      const std::size_t end = pretty_function.rfind("(") - begin;
      return pretty_function.substr(begin, end);
    }
    return "";
  }

  static inline Direction infer_direction(const char *name = nullptr) {
    const Direction dir = FrontendAttributePusher::GetDirection();
    if (dir != Direction::AUTO) {
      return dir;
    }
    if (FrontendAttributePusher::IsAutogradThread()) {
      return Direction::BWD;
    }
    return Direction::FWD;
  }

 private:

  static inline bool ends_with(std::string const & value, std::string const & ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
  }

  static inline const char *prev_char(const char *original, const char *start, char c) {
    while (start > original && *start != c) {
      --start;
    }
    return start;
  }

  static inline std::string short_fn_name(const std::string &fn_name) {
    std::string result = fn_name;
    const char *start = fn_name.c_str();
    const char *s = std::strchr(start, '(');
    if (s && *s && s > start) {
      ++s;
      if (*s) {
        if (const char *s0 = prev_char(start, s - 1, ' ')) {
          if (*s0 == ' ') {
            ++s0;
          }
          const size_t sz = s - s0 + 1;
          result = std::string(s0, sz);
          std::replace(result.begin(), result.end(), ':', '_');
        }
      }
    }
    return result;
  }
};


struct BackpropScope {
 public:
  explicit BackpropScope(const std::function<FrontendAttributePusher::FrontendAttributeContext::BackpropFor()>& fn)
      : was_enabled_(FrontendAttributePusher::IsEnabled()) {
    if (was_enabled_) {
      // TODO: make a swap call
      save_ = FrontendAttributePusher::GetBackpropOperationFor();
      FrontendAttributePusher::SetBackpropOperationFor(fn());
    }
  }
  ~BackpropScope() {
    if (was_enabled_) {
      FrontendAttributePusher::SetBackpropOperationFor(std::move(save_));
    }
  }
  const bool was_enabled_;
  FrontendAttributePusher::FrontendAttributeContext::BackpropFor save_;
};

#define PYTORCH_PTAG_INSTANTIATE_PARTITIONS(force_enable) \
    namespace pytorch_ptag { \
        PYTORCH_PTAG_PUBLIC PYTORCH_PTAG_THREAD_LOCAL FrontendAttributePusher::FrontendAttributeContext \
        FrontendAttributePusher::g_frontend_attribute_context;                                    \
        PYTORCH_PTAG_PUBLIC PYTORCH_PTAG_THREAD_LOCAL signed int FrontendAttributePusher::enable_count_{0};     \
        PYTORCH_PTAG_PUBLIC PYTORCH_PTAG_THREAD_LOCAL bool FrontendAttributePusher::force_enable_{force_enable};      \
  }  /* end of pytorch_ptag namespace */


#define PYTORCH_PTAG_ENABLE_PARITIONS_MACRO

#ifdef PYTORCH_PTAG_ENABLE_PARITIONS_MACRO
#define PYTORCH_PTAG_DECLARE_PARTITION()                    \
  pytorch_ptag::FrontendAttributePusher fattr(    \
      pytorch_ptag::PartitionScope::PartitionMatchName(\
        pytorch_ptag::PartitionScope::infer_direction(__FUNCTION__)), \
        pytorch_ptag::PartitionScope::MakePartitionName(__FUNCTION__) \
      )

#define PYTORCH_PTAG_DECLARE_PARTITION_FWD()                    \
  pytorch_ptag::FrontendAttributePusher fattr(    \
      pytorch_ptag::PartitionScope::PartitionMatchName(\
        pytorch_ptag::Direction::FWD), \
        pytorch_ptag::PartitionScope::MakePartitionName(__FUNCTION__) \
      )

#define PYTORCH_PTAG_DECLARE_PARTITION_CLASS_FWD()                    \
  pytorch_ptag::FrontendAttributePusher fattr(    \
      pytorch_ptag::PartitionScope::PartitionMatchName(\
        pytorch_ptag::Direction::BWD), \
        pytorch_ptag::PartitionScope::MakePartitionName(pytorch_ptag::PartitionScope::method_name(__PRETTY_FUNCTION__)) \
      )

#define PYTORCH_PTAG_DECLARE_PARTITION_BYNAME_FWD(__name$)                    \
  pytorch_ptag::FrontendAttributePusher fattr(    \
      pytorch_ptag::PartitionScope::PartitionMatchName(\
        pytorch_ptag::Direction::FWD), \
        pytorch_ptag::PartitionScope::MakePartitionName(__name$) \
      )

#define PYTORCH_PTAG_DECLARE_PARTITION_BWD()                    \
  pytorch_ptag::FrontendAttributePusher fattr(    \
      pytorch_ptag::PartitionScope::PartitionMatchName(\
        pytorch_ptag::Direction::BWD), \
        pytorch_ptag::PartitionScope::MakePartitionName(__FUNCTION__) \
      )

#define PYTORCH_PTAG_DECLARE_PARTITION_CLASS_BWD()                    \
  pytorch_ptag::FrontendAttributePusher fattr(    \
      pytorch_ptag::PartitionScope::PartitionMatchName(\
        pytorch_ptag::Direction::BWD), \
        pytorch_ptag::PartitionScope::MakePartitionName(\
          pytorch_ptag::PartitionScope::method_name(__PRETTY_FUNCTION__) \
      ))

#define PYTORCH_PTAG_DECLARE_PARTITION_BYNAME_BWD(__name$)                    \
  pytorch_ptag::FrontendAttributePusher fattr(    \
      pytorch_ptag::PartitionScope::PartitionMatchName(\
        pytorch_ptag::Direction::BWD), \
        pytorch_ptag::PartitionScope::MakePartitionName(__name$), \
      )

#else
#define PYTORCH_PTAG_DECLARE_PARTITION() ((void)0)
#define PYTORCH_PTAG_DECLARE_XLA_PARTITION() ((void)0)extra_attributes)
#endif

#ifndef PYTORCH_PTAG_NO_FA_SETTER
/**
 * RAII class which sets and removes a map of frontend attributes
 * to an XlaBuilder object as this class' instantiation
 * goes in and out of scope
 */
template<typename NODE_TYPE>
class FrontendAttributeSetter {
public:
    FrontendAttributeSetter(xla::XlaBuilder* builder,
                            const std::unordered_map<std::string, std::string>& attributes,
                            std::unordered_map<std::string, std::string> extra_attributes = {})
        : builder_(builder) {
        if (!attributes.empty()) {
            set_ = true;
            xla::FrontendAttributes frontend_attributes;
            frontend_attributes.CopyFrom(builder_->frontend_attributes());
            for (const auto& item : attributes) {
                frontend_attributes.mutable_map()->insert({item.first, item.second});
            }
            for (const auto& item : extra_attributes) {
                frontend_attributes.mutable_map()->insert({item.first, item.second});
            }
            save_ = builder->SwapFrontendAttributes(frontend_attributes);
        }
    }
    FrontendAttributeSetter(xla::XlaBuilder* builder, const NODE_TYPE *node)
        : FrontendAttributeSetter(builder, node->GetFrontendAttributes()) {}
    ~FrontendAttributeSetter() {
        if (set_) {
            builder_->ClearOpMetadata();
            builder_->SetFrontendAttributes(save_);
        }
    }
    std::string Dump() {
        std::stringstream ss;
        for (const auto& item : builder_->frontend_attributes().map()) {
            ss << item.first << " -> " << item.second << ", ";
        }
        return ss.str();
    }

private:
    xla::XlaBuilder* builder_;
    xla::FrontendAttributes save_;
    bool set_ = false;
};

#endif // PYTORCH_PTAG_NO_FA_SETTER

}  // end of pytorch_ptag namespace

#endif  // PYTORCH_PTAG_SCOPE_H_INCLUDED
