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
#endif

#include <cccl/c/types_official.h>
#include <string>
#include <format>

namespace cccl {
namespace jit {

//==============================================================================
// Type Info Mapping
//==============================================================================

// Type mapping template - maps C types to C++ types
template <typename T>
struct cccl_type_info_mapping
{
  using Type = T;
};

// Convert cccl_type_enum to C++ type name string
inline std::string cccl_type_enum_to_name(cccl_type_enum type) {
    switch (type) {
        case CCCL_INT8:     return "int8_t";
        case CCCL_INT16:    return "int16_t";
        case CCCL_INT32:    return "int32_t";
        case CCCL_INT64:    return "int64_t";
        case CCCL_UINT8:    return "uint8_t";
        case CCCL_UINT16:   return "uint16_t";
        case CCCL_UINT32:   return "uint32_t";
        case CCCL_UINT64:   return "uint64_t";
        case CCCL_FLOAT32:  return "float";
        case CCCL_FLOAT64:  return "double";
        case CCCL_FLOAT16:  return "__half";
        default:            return "int32_t";
    }
}

//==============================================================================
// Parameter Mapping for cccl_type_info
//==============================================================================

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS

template <>
struct parameter_mapping<cccl_type_info>
{
  static const constexpr auto archetype = cccl_type_info_mapping<int>{};

  template <typename TpId>
  static std::string map(TpId, cccl_type_info arg)
  {
    return std::format("cccl_type_info_mapping<{}>{{}}",
                       cccl_type_enum_to_name(arg.type));
  }

  template <typename TpId>
  static std::string aux(TpId, cccl_type_info)
  {
    // No auxiliary code needed for basic types
    return {};
  }
};

#endif // _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS

} // namespace jit
} // namespace cccl
