//==============================================================================
// MCRTC Compilation Utilities
// Helper functions for runtime compilation with MCRTC
//==============================================================================

#pragma once

#include <mcrtc.h>
#include <vector>
#include <string>
#include <iostream>

namespace cccl {
namespace mcrtc {

struct CompilationResult {
    bool success;
    void* bitcode;
    size_t bitcode_size;
    std::string error_log;
};

// Compile source code with MCRTC
inline CompilationResult compile_source(
    const std::string& source_code,
    const std::string& program_name,
    const std::vector<std::string>& include_paths = {},
    const std::vector<std::string>& extra_options = {}
) {
    CompilationResult result;
    result.success = false;
    result.bitcode = nullptr;
    result.bitcode_size = 0;

    // Create MCRTC program
    mcrtcProgram prog;
    mcrtcResult rtc_result = mcrtcCreateProgram(
        &prog,
        source_code.c_str(),
        program_name.c_str(),
        0, nullptr, nullptr
    );

    if (rtc_result != MCRTC_SUCCESS) {
        result.error_log = "Failed to create MCRTC program";
        return result;
    }

    // Build compile options
    std::vector<std::string> all_options;
    all_options.push_back("-xmaca");
    all_options.push_back("-std=c++17");
    all_options.push_back("--device-as-default-execution-space");

    // Add include paths
    for (const auto& path : include_paths) {
        all_options.push_back("-I" + path);
    }

    // Add extra options
    for (const auto& opt : extra_options) {
        all_options.push_back(opt);
    }

    // Convert to const char*
    std::vector<const char*> opts;
    for (const auto& opt : all_options) {
        opts.push_back(opt.c_str());
    }

    // Compile
    std::cout << "Compiling with " << opts.size() << " options..." << std::endl;
    rtc_result = mcrtcCompileProgram(prog, opts.size(), opts.data());

    if (rtc_result != MCRTC_SUCCESS) {
        // Get compilation log
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

    std::cout << "✅ Compilation successful! Bitcode size: " << result.bitcode_size << " bytes" << std::endl;

    return result;
}

// Free compilation result
inline void free_compilation_result(CompilationResult& result) {
    if (result.bitcode != nullptr) {
        delete[] static_cast<char*>(result.bitcode);
        result.bitcode = nullptr;
    }
    result.bitcode_size = 0;
}

} // namespace mcrtc
} // namespace cccl
