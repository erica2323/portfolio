//==============================================================================
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//==============================================================================

#pragma once

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
#  include "../traits.h"
#  include "type_info.h"
#endif

#include <cccl/c/types_official.h>
#include <string>
#include <format>

namespace cccl {
namespace jit {

//==============================================================================
// Operation Mapping
//==============================================================================

struct cccl_op_t_mapping
{
  bool is_stateless   = false;
  int size            = 1;
  int alignment       = 1;
  void (*operation)() = nullptr;
};

// Get operation name for code generation
inline std::string get_op_name(cccl_op_kind_t op_type) {
    switch (op_type) {
        case CCCL_PLUS:     return "cub_sum_op";
        case CCCL_MINIMUM:  return "cub_min_op";
        case CCCL_MAXIMUM:  return "cub_max_op";
        case CCCL_CUSTOM:   return "custom_op";
        case CCCL_STATELESS: return "stateless_op";
        default:            return "unknown_op";
    }
}

// Get builtin CUB operator name
inline std::string get_builtin_op_name(cccl_op_kind_t op_type) {
    switch (op_type) {
        case CCCL_PLUS:     return "::cub::Sum";
        case CCCL_MINIMUM:  return "::cub::Min";
        case CCCL_MAXIMUM:  return "::cub::Max";
        default:            return "::cub::Sum";
    }
}

//==============================================================================
// Parameter Mapping for cccl_op_t
//==============================================================================

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS

template <>
struct parameter_mapping<cccl_op_t>
{
  static const constexpr auto archetype = cccl_op_t_mapping{};

  template <typename Traits>
  static std::string map(template_id<Traits>, cccl_op_t op)
  {
    bool is_stateless = (op.type == cccl_op_kind_t::CCCL_STATELESS);

    return std::format(
      "cccl_op_t_mapping{{.is_stateless = {}, .size = {}, .alignment = {}, .operation = {}}}",
      is_stateless,
      op.size,
      op.alignment,
      op.name ? op.name : "nullptr");
  }

  template <typename Traits>
  static std::string aux(template_id<Traits>, cccl_op_t op)
  {
    // If custom operation with code, generate function declaration
    if (op.type == cccl_op_kind_t::CCCL_CUSTOM && op.code != nullptr)
    {
      return std::format(R"(
// Custom operation function
extern "C" __device__ void {}();
)",
                         op.name ? op.name : "custom_op");
    }

    // For builtin operations, no auxiliary code needed
    return {};
  }
};

#endif // _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS

} // namespace jit
} // namespace cccl
