//==============================================================================
// MACA CCCL - MCRTC Helper
// Runtime compilation wrapper for MACA (equivalent to NVIDIA's nvrtc)
//
// Based on MACA documentation:
// - Uses LLVM bitcode format
// - mcrtcGetCode returns bitcode (not PTX like CUDA)
// - mcModuleLoadData loads bitcode into GPU driver
//==============================================================================

#ifndef CCCL_MCRTC_HELPER_H
#define CCCL_MCRTC_HELPER_H

#include <mcr/mc_runtime.h>
#include <mcrtc.h>  // MACA runtime compiler header
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <iostream>

// Note: mcrtcProgram and mcrtcResult are defined in mcrtc.h
// The API matches NVIDIA's nvrtc but outputs LLVM bitcode instead of PTX

//==============================================================================
// C++ Wrapper Classes
//==============================================================================

#ifdef __cplusplus

namespace cccl {
namespace mcrtc {

/**
 * Result of a compilation operation
 */
struct CompileResult {
    std::vector<char> bitcode;   // Compiled LLVM bitcode
    std::string log;             // Compilation log
    bool success;
    mcrtcResult error_code;

    CompileResult() : success(false), error_code(MCRTC_SUCCESS) {}
};

/**
 * Result of module loading
 */
struct LoadResult {
    mcModule_t module;
    mcFunction_t kernel;
    std::string kernel_name;
    bool success;
    mcError_t error_code;

    LoadResult() : module(nullptr), kernel(nullptr), success(false), error_code(mcSuccess) {}
};

/**
 * RAII wrapper for mcrtcProgram
 */
class Program {
public:
    Program() : prog_(nullptr) {}

    ~Program() {
        if (prog_) {
            mcrtcDestroyProgram(&prog_);
        }
    }

    // Non-copyable
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;

    // Movable
    Program(Program&& other) noexcept : prog_(other.prog_) {
        other.prog_ = nullptr;
    }

    Program& operator=(Program&& other) noexcept {
        if (this != &other) {
            if (prog_) {
                mcrtcDestroyProgram(&prog_);
            }
            prog_ = other.prog_;
            other.prog_ = nullptr;
        }
        return *this;
    }

    /**
     * Create program from source code
     */
    mcrtcResult create(
        const char* src,
        const char* name = "",
        int numHeaders = 0,
        const char* const* headers = nullptr,
        const char* const* includeNames = nullptr
    ) {
        return mcrtcCreateProgram(&prog_, src, name, numHeaders, headers, includeNames);
    }

    /**
     * Compile the program
     * Default option: "-x maca" for MACA compilation mode
     */
    mcrtcResult compile(const std::vector<std::string>& options = {"-x maca"}) {
        std::vector<const char*> option_ptrs;
        for (const auto& opt : options) {
            option_ptrs.push_back(opt.c_str());
        }
        return mcrtcCompileProgram(
            prog_,
            static_cast<int>(options.size()),
            option_ptrs.empty() ? nullptr : option_ptrs.data()
        );
    }

    /**
     * Get compilation log (errors and warnings)
     */
    std::string getLog() const {
        size_t log_size = 0;
        mcrtcResult res = mcrtcGetProgramLogSize(prog_, &log_size);
        if (res != MCRTC_SUCCESS || log_size == 0) {
            return "";
        }

        std::string log(log_size, '\0');
        mcrtcGetProgramLog(prog_, &log[0]);
        return log;
    }

    /**
     * Get compiled bitcode
     */
    std::vector<char> getBitcode() const {
        size_t code_size = 0;
        mcrtcResult res = mcrtcGetCodeSize(prog_, &code_size);
        if (res != MCRTC_SUCCESS || code_size == 0) {
            return {};
        }

        std::vector<char> code(code_size);
        mcrtcGetCode(prog_, code.data());
        return code;
    }

    mcrtcProgram get() const { return prog_; }
    bool valid() const { return prog_ != nullptr; }

private:
    mcrtcProgram prog_;
};

/**
 * RAII wrapper for mcModule_t
 */
class Module {
public:
    Module() : module_(nullptr) {}

    ~Module() {
        if (module_) {
            mcModuleUnload(module_);
        }
    }

    // Non-copyable
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    // Movable
    Module(Module&& other) noexcept : module_(other.module_) {
        other.module_ = nullptr;
    }

    Module& operator=(Module&& other) noexcept {
        if (this != &other) {
            if (module_) {
                mcModuleUnload(module_);
            }
            module_ = other.module_;
            other.module_ = nullptr;
        }
        return *this;
    }

    /**
     * Load module from bitcode data
     */
    mcError_t loadData(const void* bitcode) {
        return mcModuleLoadData(&module_, bitcode);
    }

    /**
     * Get kernel function by name
     */
    mcError_t getFunction(mcFunction_t* kernel, const char* name) {
        return mcModuleGetFunction(kernel, module_, name);
    }

    mcModule_t get() const { return module_; }
    bool valid() const { return module_ != nullptr; }

    /**
     * Release ownership of the module (caller takes responsibility)
     */
    mcModule_t release() {
        mcModule_t m = module_;
        module_ = nullptr;
        return m;
    }

private:
    mcModule_t module_;
};

/**
 * High-level compiler interface
 */
class Compiler {
public:
    /**
     * Compile source code to bitcode
     */
    static CompileResult compile(
        const std::string& source,
        const std::string& name = "",
        const std::vector<std::string>& options = {"-x maca"}
    ) {
        CompileResult result;

        Program prog;
        result.error_code = prog.create(source.c_str(), name.c_str());
        if (result.error_code != MCRTC_SUCCESS) {
            result.log = "Failed to create program";
            return result;
        }

        // Compile
        result.error_code = prog.compile(options);
        result.log = prog.getLog();

        if (result.error_code != MCRTC_SUCCESS) {
            std::cerr << "[MCRTC] Compilation failed:\n" << result.log << std::endl;
            return result;
        }

        // Get compiled bitcode
        result.bitcode = prog.getBitcode();
        if (result.bitcode.empty()) {
            result.log = "Failed to get compiled bitcode";
            result.error_code = MCRTC_ERROR_INTERNAL_ERROR;
            return result;
        }

        result.success = true;
        return result;
    }

    /**
     * Compile and load module, then get kernel function
     */
    static LoadResult compileAndLoad(
        const std::string& source,
        const std::string& kernel_name,
        const std::vector<std::string>& options = {"-x maca"}
    ) {
        LoadResult result;
        result.kernel_name = kernel_name;

        // Compile
        CompileResult compile_result = compile(source, "", options);
        if (!compile_result.success) {
            result.error_code = mcErrorInvalidValue;
            return result;
        }

        // Load module
        Module module;
        result.error_code = module.loadData(compile_result.bitcode.data());
        if (result.error_code != mcSuccess) {
            std::cerr << "[MCRTC] Failed to load module, error: "
                      << result.error_code << std::endl;
            return result;
        }

        // Get kernel function
        result.error_code = module.getFunction(&result.kernel, kernel_name.c_str());
        if (result.error_code != mcSuccess) {
            std::cerr << "[MCRTC] Failed to get kernel function: " << kernel_name
                      << ", error: " << result.error_code << std::endl;
            return result;
        }

        // Transfer ownership
        result.module = module.release();
        result.success = true;
        return result;
    }
};

/**
 * Helper to launch a JIT-compiled kernel
 */
inline mcError_t launchKernel(
    mcFunction_t kernel,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    mcStream_t stream,
    void** kernelParams,
    void** extra = nullptr
) {
    return mcModuleLaunchKernel(
        kernel,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes,
        stream,
        kernelParams,
        extra
    );
}

/**
 * Helper to unload a module
 */
inline mcError_t unloadModule(mcModule_t module) {
    return mcModuleUnload(module);
}

} // namespace mcrtc
} // namespace cccl

#endif // __cplusplus

#endif // CCCL_MCRTC_HELPER_H
