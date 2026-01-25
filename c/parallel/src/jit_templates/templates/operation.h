//==============================================================================
// MACA CCCL - JIT Template: Operation
// Device code templates for user-defined operations
//==============================================================================

#ifndef CCCL_JIT_TEMPLATES_OPERATION_H
#define CCCL_JIT_TEMPLATES_OPERATION_H

// This file is designed to be embedded into JIT-compiled code
// It provides wrapper types for user-defined operations

//==============================================================================
// Stateless User Operation
//
// Wraps a user-provided device function with signature:
//   void fn(void* result, const void* arg0, const void* arg1)
//
// The operation is "stateless" - it has no per-operation data
//==============================================================================

template <typename Tag, size_t Size, size_t Alignment>
struct stateless_user_operation {
    using value_type = void;  // Type-erased

    // The device function is resolved at link time
    // User provides LTOIR with this symbol defined

    template <typename T>
    __device__ __forceinline__
    T operator()(const T& lhs, const T& rhs) const {
        // Aligned storage for result
        alignas(Alignment) char result_buf[Size];

        // Call user's device function (extern declaration must exist)
        // This function is linked from user-provided LTOIR
        extern "C" __device__ void __cccl_user_op(
            void* __restrict__ result,
            const void* __restrict__ arg0,
            const void* __restrict__ arg1
        );

        __cccl_user_op(result_buf, &lhs, &rhs);

        return *reinterpret_cast<T*>(result_buf);
    }

    // For reduce operations that need to accumulate
    template <typename T>
    __device__ __forceinline__
    T operator()(T& accumulator, const T& value) const {
        return (*this)(accumulator, value);
    }
};

//==============================================================================
// Stateful User Operation
//
// Wraps a user-provided device function with signature:
//   void fn(void* result, const void* state, const void* arg0, const void* arg1)
//
// The operation carries per-operation state
//==============================================================================

template <typename Tag, size_t Size, size_t Alignment, size_t StateSize, size_t StateAlign>
struct stateful_user_operation {
    using value_type = void;

    // Aligned state storage
    struct alignas(StateAlign) state_type {
        char data[StateSize];
    };

    state_type state;

    // Default constructor
    __host__ __device__ stateful_user_operation() = default;

    // Construct from state
    __host__ __device__ explicit stateful_user_operation(const state_type& s)
        : state(s) {}

    template <typename T>
    __device__ __forceinline__
    T operator()(const T& lhs, const T& rhs) const {
        alignas(Alignment) char result_buf[Size];

        extern "C" __device__ void __cccl_user_op_stateful(
            void* __restrict__ result,
            const void* __restrict__ state,
            const void* __restrict__ arg0,
            const void* __restrict__ arg1
        );

        __cccl_user_op_stateful(result_buf, &state, &lhs, &rhs);

        return *reinterpret_cast<T*>(result_buf);
    }
};

//==============================================================================
// Unary User Operation
//
// For operations that take a single argument (e.g., transform, negate)
//==============================================================================

template <typename Tag, size_t Size, size_t Alignment>
struct unary_user_operation {
    template <typename T>
    __device__ __forceinline__
    T operator()(const T& arg) const {
        alignas(Alignment) char result_buf[Size];

        extern "C" __device__ void __cccl_user_unary_op(
            void* __restrict__ result,
            const void* __restrict__ arg
        );

        __cccl_user_unary_op(result_buf, &arg);

        return *reinterpret_cast<T*>(result_buf);
    }
};

//==============================================================================
// Predicate User Operation
//
// For operations that return bool (e.g., filter conditions)
//==============================================================================

template <typename Tag, size_t ArgSize, size_t ArgAlign>
struct predicate_user_operation {
    template <typename T>
    __device__ __forceinline__
    bool operator()(const T& arg) const {
        bool result;

        extern "C" __device__ void __cccl_user_predicate(
            bool* __restrict__ result,
            const void* __restrict__ arg
        );

        __cccl_user_predicate(&result, &arg);

        return result;
    }
};

//==============================================================================
// Binary Predicate User Operation
//
// For comparison operations (e.g., custom sort comparators)
//==============================================================================

template <typename Tag, size_t ArgSize, size_t ArgAlign>
struct binary_predicate_user_operation {
    template <typename T>
    __device__ __forceinline__
    bool operator()(const T& lhs, const T& rhs) const {
        bool result;

        extern "C" __device__ void __cccl_user_binary_predicate(
            bool* __restrict__ result,
            const void* __restrict__ arg0,
            const void* __restrict__ arg1
        );

        __cccl_user_binary_predicate(&result, &lhs, &rhs);

        return result;
    }
};

#endif // CCCL_JIT_TEMPLATES_OPERATION_H
