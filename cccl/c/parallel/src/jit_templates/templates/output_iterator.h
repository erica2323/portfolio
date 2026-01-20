//==============================================================================
// Output Iterator Template
// For now, outputs are typically simple pointers
//==============================================================================

#pragma once

#include <cccl/c/types_official.h>
#include <string>
#include <sstream>
#include "../mappings/type_info.h"

namespace cccl {
namespace jit {

//==============================================================================
// Output Iterator Code Generation
//==============================================================================

inline std::string generate_output_iterator(
    const cccl_iterator_t& iterator,
    const std::string& iterator_name = "OutputIterator"
) {
    std::stringstream ss;
    std::string value_type = type_to_name(iterator.value_type.type);

    // For reduce operations, output is typically a simple pointer
    // More complex output iterators can be added later if needed
    ss << "// Output iterator (simple pointer)\n";
    ss << "typedef " << value_type << "* " << iterator_name << ";\n";

    return ss.str();
}

} // namespace jit
} // namespace cccl
