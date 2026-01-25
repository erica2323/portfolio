//==============================================================================
// MACA CCCL - MCJitLink Helper
// JIT Linking wrapper for MACA (equivalent to NVIDIA's nvJitLink)
//==============================================================================

#ifndef CCCL_MCJITLINK_HELPER_H
#define CCCL_MCJITLINK_HELPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// MCJitLink Types
//==============================================================================

typedef struct mcJitLinkHandle_st* mcJitLinkHandle;

typedef enum {
    MCJITLINK_SUCCESS = 0,
    MCJITLINK_ERROR_UNRECOGNIZED_OPTION = 1,
    MCJITLINK_ERROR_MISSING_ARCH = 2,
    MCJITLINK_ERROR_INVALID_INPUT = 3,
    MCJITLINK_ERROR_PTX_COMPILE = 4,
    MCJITLINK_ERROR_NVVM_COMPILE = 5,
    MCJITLINK_ERROR_INTERNAL = 6,
    MCJITLINK_ERROR_THREADPOOL = 7
} mcJitLinkResult;

typedef enum {
    MCJITLINK_INPUT_LTOIR = 0,
    MCJITLINK_INPUT_CUBIN = 1,
    MCJITLINK_INPUT_PTX = 2,
    MCJITLINK_INPUT_FATBIN = 3,
    MCJITLINK_INPUT_OBJECT = 4,
    MCJITLINK_INPUT_LIBRARY = 5
} mcJitLinkInputType;

//==============================================================================
// MCJitLink API Functions
// These should match MACA's actual JIT link API
//==============================================================================

/**
 * Create a new JIT linker handle
 */
mcJitLinkResult mcJitLinkCreate(
    mcJitLinkHandle* handle,
    uint32_t numOptions,
    const char** options
);

/**
 * Destroy a JIT linker handle
 */
mcJitLinkResult mcJitLinkDestroy(mcJitLinkHandle* handle);

/**
 * Add input data to the linker
 */
mcJitLinkResult mcJitLinkAddData(
    mcJitLinkHandle handle,
    mcJitLinkInputType inputType,
    const void* data,
    size_t size,
    const char* name
);

/**
 * Add input file to the linker
 */
mcJitLinkResult mcJitLinkAddFile(
    mcJitLinkHandle handle,
    mcJitLinkInputType inputType,
    const char* fileName
);

/**
 * Complete the linking process
 */
mcJitLinkResult mcJitLinkComplete(mcJitLinkHandle handle);

/**
 * Get the size of the linked binary
 */
mcJitLinkResult mcJitLinkGetLinkedBinarySize(
    mcJitLinkHandle handle,
    size_t* size
);

/**
 * Get the linked binary
 */
mcJitLinkResult mcJitLinkGetLinkedBinary(
    mcJitLinkHandle handle,
    void* binary
);

/**
 * Get the size of the error log
 */
mcJitLinkResult mcJitLinkGetErrorLogSize(
    mcJitLinkHandle handle,
    size_t* size
);

/**
 * Get the error log
 */
mcJitLinkResult mcJitLinkGetErrorLog(
    mcJitLinkHandle handle,
    char* log
);

/**
 * Get the size of the info log
 */
mcJitLinkResult mcJitLinkGetInfoLogSize(
    mcJitLinkHandle handle,
    size_t* size
);

/**
 * Get the info log
 */
mcJitLinkResult mcJitLinkGetInfoLog(
    mcJitLinkHandle handle,
    char* log
);

#ifdef __cplusplus
}
#endif

//==============================================================================
// C++ Wrapper
//==============================================================================

#ifdef __cplusplus

#include <string>
#include <vector>
#include <memory>

namespace cccl {
namespace mcrtc {

/**
 * Result of a linking operation
 */
struct LinkResult {
    std::vector<char> binary;    // Linked binary
    std::string error_log;       // Error messages
    std::string info_log;        // Info messages
    bool success;
    mcJitLinkResult error_code;

    LinkResult() : success(false), error_code(MCJITLINK_SUCCESS) {}
};

/**
 * RAII wrapper for mcJitLinkHandle
 */
class JitLinker {
public:
    JitLinker() : handle_(nullptr) {}

    ~JitLinker() {
        if (handle_) {
            mcJitLinkDestroy(&handle_);
        }
    }

    // Non-copyable
    JitLinker(const JitLinker&) = delete;
    JitLinker& operator=(const JitLinker&) = delete;

    // Movable
    JitLinker(JitLinker&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    JitLinker& operator=(JitLinker&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                mcJitLinkDestroy(&handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    mcJitLinkResult create(const std::vector<std::string>& options) {
        std::vector<const char*> opt_ptrs;
        for (const auto& opt : options) {
            opt_ptrs.push_back(opt.c_str());
        }
        return mcJitLinkCreate(
            &handle_,
            static_cast<uint32_t>(options.size()),
            opt_ptrs.empty() ? nullptr : opt_ptrs.data()
        );
    }

    mcJitLinkResult addLTOIR(const void* data, size_t size, const char* name) {
        return mcJitLinkAddData(handle_, MCJITLINK_INPUT_LTOIR, data, size, name);
    }

    mcJitLinkResult addPTX(const void* data, size_t size, const char* name) {
        return mcJitLinkAddData(handle_, MCJITLINK_INPUT_PTX, data, size, name);
    }

    mcJitLinkResult addCubin(const void* data, size_t size, const char* name) {
        return mcJitLinkAddData(handle_, MCJITLINK_INPUT_CUBIN, data, size, name);
    }

    mcJitLinkResult complete() {
        return mcJitLinkComplete(handle_);
    }

    std::vector<char> getBinary() const {
        size_t size = 0;
        if (mcJitLinkGetLinkedBinarySize(handle_, &size) != MCJITLINK_SUCCESS) {
            return {};
        }
        if (size == 0) return {};

        std::vector<char> binary(size);
        mcJitLinkGetLinkedBinary(handle_, binary.data());
        return binary;
    }

    std::string getErrorLog() const {
        size_t size = 0;
        if (mcJitLinkGetErrorLogSize(handle_, &size) != MCJITLINK_SUCCESS) {
            return "";
        }
        if (size == 0) return "";

        std::string log(size, '\0');
        mcJitLinkGetErrorLog(handle_, &log[0]);
        return log;
    }

    std::string getInfoLog() const {
        size_t size = 0;
        if (mcJitLinkGetInfoLogSize(handle_, &size) != MCJITLINK_SUCCESS) {
            return "";
        }
        if (size == 0) return "";

        std::string log(size, '\0');
        mcJitLinkGetInfoLog(handle_, &log[0]);
        return log;
    }

    bool valid() const { return handle_ != nullptr; }

private:
    mcJitLinkHandle handle_;
};

/**
 * High-level linker interface
 */
class Linker {
public:
    struct LTOIRInput {
        const void* data;
        size_t size;
        std::string name;
    };

    /**
     * Link multiple LTOIR inputs into a single binary
     */
    static LinkResult link(
        const std::vector<LTOIRInput>& inputs,
        const std::vector<std::string>& options
    ) {
        LinkResult result;

        JitLinker linker;
        result.error_code = linker.create(options);
        if (result.error_code != MCJITLINK_SUCCESS) {
            result.error_log = "Failed to create linker";
            return result;
        }

        // Add all inputs
        for (const auto& input : inputs) {
            if (input.size > 0) {
                result.error_code = linker.addLTOIR(
                    input.data, input.size, input.name.c_str()
                );
                if (result.error_code != MCJITLINK_SUCCESS) {
                    result.error_log = linker.getErrorLog();
                    return result;
                }
            }
        }

        // Complete linking
        result.error_code = linker.complete();
        result.error_log = linker.getErrorLog();
        result.info_log = linker.getInfoLog();

        if (result.error_code != MCJITLINK_SUCCESS) {
            return result;
        }

        result.binary = linker.getBinary();
        result.success = true;
        return result;
    }
};

} // namespace mcrtc
} // namespace cccl

#endif // __cplusplus

#endif // CCCL_MCJITLINK_HELPER_H
