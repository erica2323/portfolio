//==============================================================================
// MCRTC Compilation Utilities
// Runtime compilation with MCRTC
//==============================================================================

#pragma once

#include <mcrtc.h>
#include <vector>
#include <string>
#include <iostream>

namespace cccl {
namespace jit {

//==============================================================================
// Compilation Result
//==============================================================================

struct compilation_result {
    bool success;
    void* bitcode;
    size_t bitcode_size;
    std::string error_log;

    compilation_result() : success(false), bitcode(nullptr), bitcode_size(0) {}

    ~compilation_result() {
        if (bitcode) {
            delete[] static_cast<char*>(bitcode);
            bitcode = nullptr;
        }
    }

    // Prevent copying
    compilation_result(const compilation_result&) = delete;
    compilation_result& operator=(const compilation_result&) = delete;

    // Allow moving
    compilation_result(compilation_result&& other) noexcept
        : success(other.success)
        , bitcode(other.bitcode)
        , bitcode_size(other.bitcode_size)
        , error_log(std::move(other.error_log))
    {
        other.bitcode = nullptr;
        other.bitcode_size = 0;
    }
};

//==============================================================================
// Compilation Function
//==============================================================================

inline compilation_result compile_kernel(
    const std::string& source_code,
    const std::string& kernel_name,
    const std::vector<std::string>& include_paths = {},
    const std::vector<std::string>& extra_options = {},
    bool verbose = false
) {
    compilation_result result;

    // Create MCRTC program
    mcrtcProgram prog;
    mcrtcResult rtc_result = mcrtcCreateProgram(
        &prog,
        source_code.c_str(),
        kernel_name.c_str(),
        0, nullptr, nullptr
    );

    if (rtc_result != MCRTC_SUCCESS) {
        result.error_log = "Failed to create MCRTC program";
        return result;
    }

    // Build compile options
    std::vector<std::string> options;
    options.push_back("-xmaca");
    options.push_back("-std=c++17");
    options.push_back("--device-as-default-execution-space");

    // Add include paths
    for (const auto& path : include_paths) {
        options.push_back("-I" + path);
    }

    // Add extra options
    options.insert(options.end(), extra_options.begin(), extra_options.end());

    // Convert to C strings
    std::vector<const char*> opts;
    for (const auto& opt : options) {
        opts.push_back(opt.c_str());
    }

    if (verbose) {
        std::cout << "Compiling kernel: " << kernel_name << std::endl;
        std::cout << "Compile options (" << opts.size() << "):" << std::endl;
        for (const auto& opt : options) {
            std::cout << "  " << opt << std::endl;
        }
    }

    // Compile
    rtc_result = mcrtcCompileProgram(prog, opts.size(), opts.data());

    if (rtc_result != MCRTC_SUCCESS) {
        // Get error log
        size_t log_size;
        mcrtcGetProgramLogSize(prog, &log_size);
        if (log_size > 1) {
            char* log = new char[log_size];
            mcrtcGetProgramLog(prog, log);
            result.error_log = std::string(log);
            delete[] log;
        }
        mcrtcDestroyProgram(&prog);
        return result;
    }

    // Get bitcode
    mcrtcGetBitcodeSize(prog, &result.bitcode_size);
    char* bitcode = new char[result.bitcode_size];
    mcrtcGetBitcode(prog, bitcode);

    result.bitcode = bitcode;
    result.success = true;

    mcrtcDestroyProgram(&prog);

    if (verbose) {
        std::cout << "✅ Compilation successful!" << std::endl;
        std::cout << "   Bitcode size: " << result.bitcode_size << " bytes" << std::endl;
    }

    return result;
}

} // namespace jit
} // namespace cccl
