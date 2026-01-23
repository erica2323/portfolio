//==============================================================================
// MACA CCCL JIT Compiler - Implementation
//==============================================================================

#include "jit_compiler.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <openssl/sha.h>  // For cache key hashing

// MCRTC API (类似 NVRTC)
// 注意：实际的头文件名可能是 mcrtc.h 或 mc_rtc.h，需要根据你的 MACA SDK 调整
#ifdef __MACA__
  // MACA 可能使用不同的 RTC 头文件
  // #include <mcrtc/mcrtc.h>
  // 如果没有专门的 MCRTC，可能需要使用 mcModule/mcFunction API
  #include <mcr/mc_runtime.h>

  // 假设的 MCRTC API（需要根据实际 MACA SDK 调整）
  typedef void* mcrtcProgram;

  extern "C" {
    // 这些函数签名需要根据你的 MACA SDK 文档调整
    mcError_t mcrtcCreateProgram(mcrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames);
    mcError_t mcrtcCompileProgram(mcrtcProgram prog, int numOptions, const char** options);
    mcError_t mcrtcGetPTXSize(mcrtcProgram prog, size_t* ptxSizeRet);
    mcError_t mcrtcGetPTX(mcrtcProgram prog, char* ptx);
    mcError_t mcrtcDestroyProgram(mcrtcProgram* prog);
    mcError_t mcrtcGetProgramLogSize(mcrtcProgram prog, size_t* logSizeRet);
    mcError_t mcrtcGetProgramLog(mcrtcProgram prog, char* log);

    mcError_t mcModuleLoadData(void** module, const void* image);
    mcError_t mcModuleGetFunction(void** kernel, void* module, const char* name);
    mcError_t mcModuleUnload(void* module);
  }
#else
  #include <nvrtc.h>
  #include <cuda.h>
  typedef nvrtcProgram mcrtcProgram;
  #define mcrtcCreateProgram nvrtcCreateProgram
  #define mcrtcCompileProgram nvrtcCompileProgram
  #define mcrtcGetPTXSize nvrtcGetPTXSize
  #define mcrtcGetPTX nvrtcGetPTX
  #define mcrtcDestroyProgram nvrtcDestroyProgram
  #define mcrtcGetProgramLogSize nvrtcGetProgramLogSize
  #define mcrtcGetProgramLog nvrtcGetProgramLog
  #define mcModuleLoadData cuModuleLoadData
  #define mcModuleGetFunction cuModuleGetFunction
  #define mcModuleUnload cuModuleUnload
#endif

using namespace cccl::jit;

//==============================================================================
// Kernel Cache Implementation
//==============================================================================

KernelCache& KernelCache::instance() {
    static KernelCache cache;
    return cache;
}

cccl_jit_kernel_t* KernelCache::get(const std::string& key) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        std::cout << "[JIT Cache] HIT: " << key.substr(0, 16) << "..." << std::endl;
        return it->second.get();
    }
    std::cout << "[JIT Cache] MISS: " << key.substr(0, 16) << "..." << std::endl;
    return nullptr;
}

void KernelCache::put(const std::string& key, cccl_jit_kernel_t* kernel) {
    kernel->is_cached = true;
    cache_[key] = std::unique_ptr<cccl_jit_kernel_t>(kernel);
    std::cout << "[JIT Cache] ADD: " << key.substr(0, 16) << "... (total: " << cache_.size() << ")" << std::endl;
}

void KernelCache::clear() {
    cache_.clear();
    std::cout << "[JIT Cache] CLEARED" << std::endl;
}

std::string KernelCache::compute_cache_key(
    const cccl_op_t* op,
    const cccl_type_info* value_type,
    int cc_major,
    int cc_minor
) {
    std::ostringstream oss;

    // Hash input: op_type + code + type + compute capability
    oss << static_cast<int>(op->type) << "|";
    if (op->code != nullptr) {
        oss << std::string(op->code, op->code_size) << "|";
    }
    oss << static_cast<int>(value_type->type) << "|";
    oss << cc_major << "." << cc_minor;

    std::string input = oss.str();

    // SHA256 hash
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(input.c_str()), input.size(), hash);

    // Convert to hex string
    std::ostringstream hex_oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        hex_oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }

    return hex_oss.str();
}

//==============================================================================
// MCRTC Compiler Implementation
//==============================================================================

mcError_t MCRTCCompiler::compile(
    const std::string& source_code,
    const std::string& kernel_name,
    const std::vector<std::string>& include_paths,
    const std::vector<std::string>& compile_options,
    std::string& ptx_out
) {
    std::cout << "[MCRTC] Compiling kernel: " << kernel_name << std::endl;
    std::cout << "[MCRTC] Source code size: " << source_code.size() << " bytes" << std::endl;

    mcrtcProgram prog;
    mcError_t err;

    // Create program
    err = mcrtcCreateProgram(
        &prog,
        source_code.c_str(),
        kernel_name.c_str(),
        0,           // numHeaders
        nullptr,     // headers
        nullptr      // includeNames
    );

    if (err != mcSuccess) {
        std::cerr << "[MCRTC] Failed to create program: " << err << std::endl;
        return err;
    }

    // Prepare compile options
    std::vector<const char*> opts;
    for (const auto& opt : compile_options) {
        opts.push_back(opt.c_str());
    }
    for (const auto& path : include_paths) {
        opts.push_back(path.c_str());
    }

    // Compile
    err = mcrtcCompileProgram(prog, opts.size(), opts.data());

    // Get compilation log (even if successful, may have warnings)
    size_t log_size = 0;
    mcrtcGetProgramLogSize(prog, &log_size);
    if (log_size > 1) {
        std::vector<char> log(log_size);
        mcrtcGetProgramLog(prog, log.data());
        std::cout << "[MCRTC] Compilation log:\n" << log.data() << std::endl;
    }

    if (err != mcSuccess) {
        std::cerr << "[MCRTC] Compilation failed: " << err << std::endl;
        mcrtcDestroyProgram(&prog);
        return err;
    }

    // Get PTX/LLVM-IR
    size_t ptx_size = 0;
    err = mcrtcGetPTXSize(prog, &ptx_size);
    if (err != mcSuccess) {
        std::cerr << "[MCRTC] Failed to get PTX size: " << err << std::endl;
        mcrtcDestroyProgram(&prog);
        return err;
    }

    std::vector<char> ptx(ptx_size);
    err = mcrtcGetPTX(prog, ptx.data());
    if (err != mcSuccess) {
        std::cerr << "[MCRTC] Failed to get PTX: " << err << std::endl;
        mcrtcDestroyProgram(&prog);
        return err;
    }

    ptx_out = std::string(ptx.data(), ptx_size);
    std::cout << "[MCRTC] ✅ Compilation successful (PTX size: " << ptx_size << " bytes)" << std::endl;

    mcrtcDestroyProgram(&prog);
    return mcSuccess;
}

mcError_t MCRTCCompiler::load_module(
    const std::string& ptx_code,
    void** module_out,
    void** kernel_func_out,
    const char* kernel_name
) {
    std::cout << "[MCRTC] Loading module..." << std::endl;

    mcError_t err = mcModuleLoadData(module_out, ptx_code.c_str());
    if (err != mcSuccess) {
        std::cerr << "[MCRTC] Failed to load module: " << err << std::endl;
        return err;
    }

    err = mcModuleGetFunction(kernel_func_out, *module_out, kernel_name);
    if (err != mcSuccess) {
        std::cerr << "[MCRTC] Failed to get kernel function: " << err << std::endl;
        mcModuleUnload(*module_out);
        return err;
    }

    std::cout << "[MCRTC] ✅ Module loaded successfully" << std::endl;
    return mcSuccess;
}

//==============================================================================
// Kernel Source Code Generator
//==============================================================================

std::string ReduceKernelGenerator::get_type_string(cccl_type_enum type) {
    switch (type) {
        case CCCL_INT32:   return "int32_t";
        case CCCL_INT64:   return "int64_t";
        case CCCL_UINT32:  return "uint32_t";
        case CCCL_UINT64:  return "uint64_t";
        case CCCL_FLOAT32: return "float";
        case CCCL_FLOAT64: return "double";
        default:           return "void";
    }
}

std::string ReduceKernelGenerator::wrap_user_op(const cccl_op_t* op) {
    if (op->code == nullptr || op->code_size == 0) {
        return "/* No user code provided */";
    }

    std::ostringstream oss;

    if (op->code_type == CCCL_OP_CPP_SOURCE) {
        // User provided C++ lambda or functor source
        oss << "// User-provided operator code:\n";
        oss << std::string(op->code, op->code_size) << "\n";
    } else if (op->code_type == CCCL_OP_LTOIR) {
        // User provided pre-compiled LTOIR (not common for custom ops)
        oss << "// LTOIR not directly embeddable in source\n";
        oss << "// This path requires linking pre-compiled bitcode\n";
    }

    return oss.str();
}

std::string ReduceKernelGenerator::generate_source(
    const cccl_op_t* op,
    const cccl_type_info* value_type,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path
) {
    std::ostringstream source;

    std::string type_str = get_type_string(value_type->type);
    std::string op_name = (op->name != nullptr) ? op->name : "CustomOp";

    source << R"(
//==============================================================================
// Auto-generated CCCL Reduce Kernel (MACA)
//==============================================================================

#include <cub/device/device_reduce.cuh>
#include <cub/util_ptx.cuh>

)";

    // Embed user operator code
    source << wrap_user_op(op) << "\n\n";

    // Generate wrapper operator struct
    source << "struct " << op_name << "_Wrapper {\n";
    source << "    __device__ __forceinline__ " << type_str << " operator()(\n";
    source << "        const " << type_str << "& a,\n";
    source << "        const " << type_str << "& b\n";
    source << "    ) const {\n";

    // Call user operator
    if (op->code != nullptr && op->code_type == CCCL_OP_CPP_SOURCE) {
        // Assume user code defines "user_op" lambda
        source << "        auto user_op = " << std::string(op->code, op->code_size) << ";\n";
        source << "        return user_op(a, b);\n";
    } else {
        source << "        return a;  // Placeholder\n";
    }

    source << "    }\n";
    source << "};\n\n";

    // Generate dispatch wrapper function
    source << "extern \"C\" __global__ void cccl_reduce_kernel(\n";
    source << "    const " << type_str << "* d_in,\n";
    source << "    " << type_str << "* d_out,\n";
    source << "    int num_items,\n";
    source << "    " << type_str << " init_value\n";
    source << ") {\n";
    source << "    // This is a placeholder - actual CUB dispatch happens in host code\n";
    source << "    // This kernel is mainly for JIT compilation testing\n";
    source << "    if (threadIdx.x == 0 && blockIdx.x == 0) {\n";
    source << "        " << type_str << " sum = init_value;\n";
    source << "        " << op_name << "_Wrapper op;\n";
    source << "        for (int i = 0; i < num_items; i++) {\n";
    source << "            sum = op(sum, d_in[i]);\n";
    source << "        }\n";
    source << "        *d_out = sum;\n";
    source << "    }\n";
    source << "}\n";

    return source.str();
}

//==============================================================================
// Public C API Implementation
//==============================================================================

extern "C" {

mcError_t cccl_jit_compile_reduce_op(
    const cccl_op_t* op,
    const cccl_type_info* value_type,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    cccl_jit_kernel_t** kernel_out
) {
    if (op == nullptr || value_type == nullptr || kernel_out == nullptr) {
        return mcErrorInvalidValue;
    }

    try {
        // Check cache first
        std::string cache_key = KernelCache::instance().compute_cache_key(
            op, value_type, cc_major, cc_minor
        );

        cccl_jit_kernel_t* cached = KernelCache::instance().get(cache_key);
        if (cached != nullptr) {
            *kernel_out = cached;
            return mcSuccess;
        }

        // Generate source code
        std::string source = ReduceKernelGenerator::generate_source(
            op, value_type, cub_path, thrust_path, libcudacxx_path
        );

        // Prepare compile options
        std::vector<std::string> include_paths;
        if (cub_path && strlen(cub_path) > 0) {
            include_paths.push_back(std::string("-I") + cub_path);
        }
        if (thrust_path && strlen(thrust_path) > 0) {
            include_paths.push_back(std::string("-I") + thrust_path);
        }
        if (libcudacxx_path && strlen(libcudacxx_path) > 0) {
            include_paths.push_back(std::string("-I") + libcudacxx_path);
        }

        std::vector<std::string> compile_options = {
            "-std=c++17",
            "-D__MACA__",
            std::string("-arch=compute_") + std::to_string(cc_major) + std::to_string(cc_minor),
            "--device-as-default-execution-space",
        };

        // Compile
        std::string ptx_code;
        mcError_t err = MCRTCCompiler::compile(
            source,
            "cccl_reduce_kernel",
            include_paths,
            compile_options,
            ptx_code
        );

        if (err != mcSuccess) {
            return err;
        }

        // Load module
        cccl_jit_kernel_t* kernel = new cccl_jit_kernel_t();
        kernel->ptx_code = new std::string(ptx_code);
        kernel->shared_mem_bytes = 0;
        kernel->is_cached = false;

        err = MCRTCCompiler::load_module(
            ptx_code,
            &kernel->module,
            &kernel->kernel_func,
            "cccl_reduce_kernel"
        );

        if (err != mcSuccess) {
            delete kernel->ptx_code;
            delete kernel;
            return err;
        }

        // Add to cache
        KernelCache::instance().put(cache_key, kernel);

        *kernel_out = kernel;
        return mcSuccess;
    }
    catch (const std::exception& e) {
        std::cerr << "[JIT] Exception: " << e.what() << std::endl;
        return mcErrorUnknown;
    }
}

mcError_t cccl_jit_kernel_destroy(cccl_jit_kernel_t* kernel) {
    if (kernel == nullptr) {
        return mcSuccess;
    }

    // Don't destroy cached kernels
    if (kernel->is_cached) {
        return mcSuccess;
    }

    try {
        if (kernel->module != nullptr) {
            mcModuleUnload(kernel->module);
        }
        if (kernel->ptx_code != nullptr) {
            delete kernel->ptx_code;
        }
        delete kernel;
        return mcSuccess;
    }
    catch (...) {
        return mcErrorUnknown;
    }
}

cccl_jit_kernel_t* cccl_jit_get_cached_kernel(const char* cache_key) {
    return KernelCache::instance().get(cache_key);
}

void cccl_jit_cache_kernel(const char* cache_key, cccl_jit_kernel_t* kernel) {
    KernelCache::instance().put(cache_key, kernel);
}

void cccl_jit_clear_cache() {
    KernelCache::instance().clear();
}

} // extern "C"
