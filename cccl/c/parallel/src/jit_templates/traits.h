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

#include <string>
#include <type_traits>

namespace cccl {
namespace jit {

//==============================================================================
// Template ID Helper
//==============================================================================

// Helper for passing trait types as template parameters
template <typename Traits>
struct template_id
{
  using type = Traits;
};

//==============================================================================
// Parameter Mapping Base Template
//==============================================================================

// Base template for parameter_mapping
// Each type (cccl_type_info, cccl_op_t, cccl_iterator_t) will specialize this
template <typename T>
struct parameter_mapping;

// parameter_mapping specializations provide:
// - archetype: A constexpr instance used for type deduction
// - map(TpId, T arg): Generates format string for template parameter
// - aux(TpId, T arg): Generates auxiliary code (function declarations, etc.)

} // namespace jit
} // namespace cccl
