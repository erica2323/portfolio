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
// Iterator Mapping
//==============================================================================

template <typename ValueTp>
struct cccl_iterator_t_mapping
{
  bool is_pointer     = false;
  int size            = 1;
  int alignment       = 1;
  void (*advance)(void*, cuda::std::uint64_t) = nullptr;
  ValueTp (*dereference)(const void*)         = nullptr;
  void (*assign)(const void*, ValueTp)        = nullptr;

  using ValueT = ValueTp;
};

// Trait tags for iterator types
struct input_iterator_traits;
struct output_iterator_traits;

//==============================================================================
// Parameter Mapping for cccl_iterator_t
//==============================================================================

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS

template <>
struct parameter_mapping<cccl_iterator_t>
{
  static const constexpr auto archetype = cccl_iterator_t_mapping<int>{};

  template <typename Traits>
  static std::string map(template_id<Traits>, cccl_iterator_t arg)
  {
    bool is_pointer = (arg.type == cccl_iterator_kind_t::CCCL_POINTER);
    std::string value_type = cccl_type_enum_to_name(arg.value_type.type);

    // Determine which function pointer field to use
    std::string func_field;
    std::string func_name;

    if constexpr (std::is_same_v<Traits, output_iterator_traits>)
    {
      func_field = "assign";
      func_name = arg.assign.name ? arg.assign.name : "nullptr";
    }
    else
    {
      func_field = "dereference";
      func_name = arg.dereference.name ? arg.dereference.name : "nullptr";
    }

    return std::format(
      "cccl_iterator_t_mapping<{}>{{.is_pointer = {}, .size = {}, .alignment = {}, .advance = {}, .{} = {}}}",
      value_type,
      is_pointer,
      arg.size,
      arg.alignment,
      arg.advance.name ? arg.advance.name : "nullptr",
      func_field,
      func_name);
  }

  template <typename Traits>
  static std::string aux(template_id<Traits>, cccl_iterator_t arg)
  {
    std::string value_type = cccl_type_enum_to_name(arg.value_type.type);
    std::string advance_name = arg.advance.name ? arg.advance.name : "advance_func";

    if constexpr (std::is_same_v<Traits, output_iterator_traits>)
    {
      // Output iterator: needs advance + assign
      std::string assign_name = arg.assign.name ? arg.assign.name : "assign_func";

      return std::format(R"output(
// Output iterator auxiliary functions
extern "C" __device__ void {0}(void *, {1});
extern "C" __device__ void {2}(const void *, {3});
)output",
                         advance_name,
                         cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64),
                         assign_name,
                         value_type);
    }
    else
    {
      // Input iterator: needs advance + dereference
      std::string deref_name = arg.dereference.name ? arg.dereference.name : "deref_func";

      return std::format(R"input(
// Input iterator auxiliary functions
extern "C" __device__ void {0}(void *, {1});
extern "C" __device__ {2} {3}(const void *);
)input",
                         advance_name,
                         cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64),
                         value_type,
                         deref_name);
    }
  }
};

#endif // _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS

} // namespace jit
} // namespace cccl
