//==============================================================================
// Iterator Mapping for JIT Templates
// Complete iterator type system matching NVIDIA CCCL
//==============================================================================

#pragma once

#include <cccl/c/types_official.h>
#include <string>
#include <sstream>
#include "type_info.h"

namespace cccl {
namespace jit {

//==============================================================================
// Iterator State Information
//==============================================================================

struct iterator_state_info {
    std::string type_name;
    size_t size;
    size_t alignment;
    bool has_state;
};

// Get iterator state info
inline iterator_state_info get_iterator_state_info(const cccl_iterator_t& it) {
    iterator_state_info info;
    info.type_name = type_to_name(it.value_type.type);
    info.size = it.value_type.size;
    info.alignment = it.value_type.alignment;

    switch (it.type) {
        case CCCL_POINTER:
            // Simple pointer - state is just the pointer
            info.has_state = true;
            info.size = sizeof(void*);
            info.alignment = alignof(void*);
            break;

        case CCCL_ITERATOR:
            // Custom iterator - may have state
            info.has_state = true;
            // State size determined by iterator implementation
            break;

        default:
            info.has_state = false;
            break;
    }

    return info;
}

//==============================================================================
// Iterator Specialization Detection
//==============================================================================

struct iterator_specialization {
    bool is_pointer;
    bool has_custom_code;
    std::string code;
};

inline iterator_specialization detect_specialization(const cccl_iterator_t& it) {
    iterator_specialization spec;
    spec.is_pointer = (it.type == CCCL_POINTER);
    spec.has_custom_code = (it.dereference.code != nullptr && strlen(it.dereference.code) > 0);
    spec.code = spec.has_custom_code ? it.dereference.code : "";
    return spec;
}

//==============================================================================
// Iterator Traits Generation
//==============================================================================

// Generate iterator traits (matching C++ iterator requirements)
inline std::string generate_iterator_traits(
    const std::string& iterator_name,
    const std::string& value_type,
    const std::string& tag = "::cub::detail::random_access_iterator_tag"
) {
    std::stringstream ss;

    ss << "    // Iterator traits (C++ standard)\n";
    ss << "    using iterator_category = " << tag << ";\n";
    ss << "    using difference_type   = ::cub::detail::size_t;\n";
    ss << "    using value_type        = " << value_type << ";\n";
    ss << "    using reference         = value_type&;\n";
    ss << "    using pointer           = value_type*;\n";

    return ss.str();
}

//==============================================================================
// Iterator Operators Generation
//==============================================================================

// Generate full set of iterator operators
inline std::string generate_iterator_operators(
    const std::string& value_type,
    bool has_custom_deref,
    const std::string& deref_impl
) {
    std::stringstream ss;

    // operator* (dereference)
    ss << "    __device__ __forceinline__\n";
    ss << "    value_type operator*() const {\n";
    if (has_custom_deref) {
        ss << "        " << deref_impl << "\n";
    } else {
        ss << "        return *ptr;\n";
    }
    ss << "    }\n\n";

    // operator[] (index access)
    ss << "    __device__ __forceinline__\n";
    ss << "    value_type operator[](difference_type idx) const {\n";
    if (has_custom_deref) {
        ss << "        auto temp = *this;\n";
        ss << "        temp += idx;\n";
        ss << "        return *temp;\n";
    } else {
        ss << "        return ptr[idx];\n";
    }
    ss << "    }\n\n";

    // operator+= (advance)
    ss << "    __device__ __forceinline__\n";
    ss << "    iterator_type& operator+=(difference_type diff) {\n";
    ss << "        ptr += diff;\n";
    ss << "        return *this;\n";
    ss << "    }\n\n";

    // operator+ (offset)
    ss << "    __device__ __forceinline__\n";
    ss << "    iterator_type operator+(difference_type diff) const {\n";
    ss << "        iterator_type result = *this;\n";
    ss << "        result += diff;\n";
    ss << "        return result;\n";
    ss << "    }\n\n";

    // operator-= (retreat)
    ss << "    __device__ __forceinline__\n";
    ss << "    iterator_type& operator-=(difference_type diff) {\n";
    ss << "        ptr -= diff;\n";
    ss << "        return *this;\n";
    ss << "    }\n\n";

    // operator- (reverse offset)
    ss << "    __device__ __forceinline__\n";
    ss << "    iterator_type operator-(difference_type diff) const {\n";
    ss << "        iterator_type result = *this;\n";
    ss << "        result -= diff;\n";
    ss << "        return result;\n";
    ss << "    }\n\n";

    // operator- (difference)
    ss << "    __device__ __forceinline__\n";
    ss << "    difference_type operator-(const iterator_type& other) const {\n";
    ss << "        return ptr - other.ptr;\n";
    ss << "    }\n";

    return ss.str();
}

} // namespace jit
} // namespace cccl
