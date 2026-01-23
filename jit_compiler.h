//==============================================================================
// MACA CCCL JIT Compiler - Header
// Provides runtime kernel compilation using MCRTC
//==============================================================================

#ifndef CCCL_C_JIT_COMPILER_H
#define CCCL_C_JIT_COMPILER_H

#include <cccl/c/types_official.h>
#include <mcr/mc_runtime.h>
#include <string>
#include <unordered_map>
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// Compiled Kernel Handle
//==============================================================================

typedef struct cccl_jit_kernel_t {
    void* kernel_func;              // mcFunction_t (compiled kernel function)
    void* module;                   // mcModule_t (loaded module)
    std::string* ptx_code;          // Generated PTX/LLVM-IR code
    size_t shared_mem_bytes;        // Required shared memory
    bool is_cached;                 // Whether this kernel is from cache
} cccl_jit_kernel_t;

//==============================================================================
// JIT Compilation Functions
//==============================================================================

/**
 * Compile a custom reduce operator using MCRTC
 *
 * @param op User-provided operator (must have code field set)
 * @param value_type Data type for reduction
 * @param cc_major Compute capability major version
 * @param cc_minor Compute capability minor version
 * @param cub_path Path to CUB headers
 * @param thrust_path Path to Thrust headers
 * @param libcudacxx_path Path to libcu++ headers
 * @param kernel_out Output compiled kernel handle
 * @return mcError_t Error code
 */
mcError_t cccl_jit_compile_reduce_op(
    const cccl_op_t* op,
    const cccl_type_info* value_type,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    cccl_jit_kernel_t** kernel_out
);

/**
 * Destroy compiled kernel and free resources
 */
mcError_t cccl_jit_kernel_destroy(cccl_jit_kernel_t* kernel);

/**
 * Get cached kernel by hash (returns nullptr if not found)
 */
cccl_jit_kernel_t* cccl_jit_get_cached_kernel(const char* cache_key);

/**
 * Add kernel to cache
 */
void cccl_jit_cache_kernel(const char* cache_key, cccl_jit_kernel_t* kernel);

/**
 * Clear all cached kernels
 */
void cccl_jit_clear_cache();

//==============================================================================
// C++ Helper Classes (internal use)
//==============================================================================

#ifdef __cplusplus
}

namespace cccl {
namespace jit {

/**
 * Kernel cache manager (singleton)
 */
class KernelCache {
public:
    static KernelCache& instance();

    cccl_jit_kernel_t* get(const std::string& key);
    void put(const std::string& key, cccl_jit_kernel_t* kernel);
    void clear();

    std::string compute_cache_key(
        const cccl_op_t* op,
        const cccl_type_info* value_type,
        int cc_major,
        int cc_minor
    );

private:
    KernelCache() = default;
    std::unordered_map<std::string, std::unique_ptr<cccl_jit_kernel_t>> cache_;
};

/**
 * MCRTC compiler wrapper
 */
class MCRTCCompiler {
public:
    /**
     * Compile source code to LLVM-IR/PTX
     *
     * @param source_code C++ source code
     * @param kernel_name Kernel function name
     * @param include_paths List of include directories
     * @param compile_options Additional compiler options
     * @param ptx_out Output PTX/IR code
     * @return mcError_t
     */
    static mcError_t compile(
        const std::string& source_code,
        const std::string& kernel_name,
        const std::vector<std::string>& include_paths,
        const std::vector<std::string>& compile_options,
        std::string& ptx_out
    );

    /**
     * Load compiled PTX into a module
     */
    static mcError_t load_module(
        const std::string& ptx_code,
        void** module_out,
        void** kernel_func_out,
        const char* kernel_name
    );
};

/**
 * Kernel source code generator
 */
class ReduceKernelGenerator {
public:
    /**
     * Generate complete reduce kernel source code
     *
     * @param op User operator
     * @param value_type Data type
     * @param cub_path CUB include path
     * @param thrust_path Thrust include path
     * @param libcudacxx_path libcu++ include path
     * @return Generated source code
     */
    static std::string generate_source(
        const cccl_op_t* op,
        const cccl_type_info* value_type,
        const char* cub_path,
        const char* thrust_path,
        const char* libcudacxx_path
    );

private:
    static std::string get_type_string(cccl_type_enum type);
    static std::string wrap_user_op(const cccl_op_t* op);
};

} // namespace jit
} // namespace cccl

#endif // __cplusplus

#endif // CCCL_C_JIT_COMPILER_H
