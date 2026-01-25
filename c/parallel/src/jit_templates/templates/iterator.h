//==============================================================================
// MACA CCCL - JIT Template: Iterator
// Device code templates for custom iterators
//==============================================================================

#ifndef CCCL_JIT_TEMPLATES_ITERATOR_H
#define CCCL_JIT_TEMPLATES_ITERATOR_H

#include <cstddef>
#include <cstdint>

//==============================================================================
// Iterator State Storage
//
// Provides aligned storage for iterator state that gets passed between
// host and device
//==============================================================================

template <typename Tag, size_t Size, size_t Alignment>
struct alignas(Alignment) input_iterator_state_t {
    char data[Size];

    __host__ __device__ input_iterator_state_t() = default;

    __host__ __device__ input_iterator_state_t(const input_iterator_state_t& other) {
        for (size_t i = 0; i < Size; ++i) {
            data[i] = other.data[i];
        }
    }

    __host__ __device__ input_iterator_state_t& operator=(const input_iterator_state_t& other) {
        for (size_t i = 0; i < Size; ++i) {
            data[i] = other.data[i];
        }
        return *this;
    }
};

//==============================================================================
// Input Iterator Template
//
// Wraps user-provided advance and dereference operations.
// The operations are linked from user-provided LTOIR/bitcode.
//
// Device function signatures:
//   void advance(void* state, int64_t offset)
//   void dereference(ValueT* result, const void* state)
//==============================================================================

template <typename Tag, size_t Size, size_t Alignment, typename ValueT>
struct input_iterator_t {
    using value_type = ValueT;
    using difference_type = int64_t;
    using pointer = ValueT*;
    using reference = ValueT&;
    using iterator_category = void;  // Random access, but custom type

    using state_type = input_iterator_state_t<Tag, Size, Alignment>;
    state_type state;

    // Default constructor
    __host__ __device__ input_iterator_t() = default;

    // Construct from state
    __host__ __device__ explicit input_iterator_t(const state_type& s)
        : state(s) {}

    // Construct from void pointer to state
    __host__ __device__ static input_iterator_t from_state(const void* state_ptr) {
        input_iterator_t it;
        const char* src = static_cast<const char*>(state_ptr);
        for (size_t i = 0; i < Size; ++i) {
            it.state.data[i] = src[i];
        }
        return it;
    }

    // Dereference operator - calls user's device function
    __device__ __forceinline__
    value_type operator*() const {
        value_type result;

        // This function is linked from user-provided bitcode
        extern "C" __device__ void __cccl_iterator_dereference(
            void* __restrict__ result,
            const void* __restrict__ state
        );

        __cccl_iterator_dereference(&result, &state);
        return result;
    }

    // Subscript operator
    __device__ __forceinline__
    value_type operator[](difference_type n) const {
        input_iterator_t tmp = *this;
        tmp += n;
        return *tmp;
    }

    // Increment operators
    __device__ __forceinline__
    input_iterator_t& operator++() {
        *this += 1;
        return *this;
    }

    __device__ __forceinline__
    input_iterator_t operator++(int) {
        input_iterator_t tmp = *this;
        ++(*this);
        return tmp;
    }

    // Decrement operators
    __device__ __forceinline__
    input_iterator_t& operator--() {
        *this -= 1;
        return *this;
    }

    __device__ __forceinline__
    input_iterator_t operator--(int) {
        input_iterator_t tmp = *this;
        --(*this);
        return tmp;
    }

    // Compound assignment with offset
    __device__ __forceinline__
    input_iterator_t& operator+=(difference_type n) {
        // This function is linked from user-provided bitcode
        extern "C" __device__ void __cccl_iterator_advance(
            void* __restrict__ state,
            int64_t offset
        );

        __cccl_iterator_advance(&state, n);
        return *this;
    }

    __device__ __forceinline__
    input_iterator_t& operator-=(difference_type n) {
        return *this += (-n);
    }

    // Arithmetic operators
    __device__ __forceinline__
    input_iterator_t operator+(difference_type n) const {
        input_iterator_t tmp = *this;
        tmp += n;
        return tmp;
    }

    __device__ __forceinline__
    input_iterator_t operator-(difference_type n) const {
        input_iterator_t tmp = *this;
        tmp -= n;
        return tmp;
    }

    // Friend function for reverse addition
    __device__ __forceinline__
    friend input_iterator_t operator+(difference_type n, const input_iterator_t& it) {
        return it + n;
    }
};

//==============================================================================
// Output Iterator Template
//
// For writing results. Simpler than input iterator.
//
// Device function signature:
//   void store(void* state, const ValueT* value)
//==============================================================================

template <typename Tag, size_t Size, size_t Alignment, typename ValueT>
struct output_iterator_t {
    using value_type = ValueT;
    using difference_type = int64_t;
    using pointer = ValueT*;
    using reference = ValueT&;

    using state_type = input_iterator_state_t<Tag, Size, Alignment>;
    state_type state;

    __host__ __device__ output_iterator_t() = default;

    __host__ __device__ explicit output_iterator_t(const state_type& s)
        : state(s) {}

    __host__ __device__ static output_iterator_t from_state(const void* state_ptr) {
        output_iterator_t it;
        const char* src = static_cast<const char*>(state_ptr);
        for (size_t i = 0; i < Size; ++i) {
            it.state.data[i] = src[i];
        }
        return it;
    }

    // Proxy for writing
    struct proxy {
        output_iterator_t* it;

        __device__ proxy& operator=(const value_type& value) {
            extern "C" __device__ void __cccl_iterator_store(
                void* __restrict__ state,
                const void* __restrict__ value
            );

            __cccl_iterator_store(&(it->state), &value);
            return *this;
        }
    };

    __device__ __forceinline__
    proxy operator*() {
        return proxy{this};
    }

    __device__ __forceinline__
    output_iterator_t& operator++() {
        extern "C" __device__ void __cccl_iterator_advance(
            void* __restrict__ state,
            int64_t offset
        );

        __cccl_iterator_advance(&state, 1);
        return *this;
    }

    __device__ __forceinline__
    output_iterator_t& operator+=(difference_type n) {
        extern "C" __device__ void __cccl_iterator_advance(
            void* __restrict__ state,
            int64_t offset
        );

        __cccl_iterator_advance(&state, n);
        return *this;
    }

    __device__ __forceinline__
    output_iterator_t operator+(difference_type n) const {
        output_iterator_t tmp = *this;
        tmp += n;
        return tmp;
    }
};

//==============================================================================
// Counting Iterator
//
// Built-in iterator that generates sequential values
//==============================================================================

template <typename T>
struct counting_iterator_t {
    using value_type = T;
    using difference_type = int64_t;
    using pointer = T*;
    using reference = T;

    T value;

    __host__ __device__ counting_iterator_t() : value(T(0)) {}
    __host__ __device__ explicit counting_iterator_t(T v) : value(v) {}

    __device__ __forceinline__
    value_type operator*() const { return value; }

    __device__ __forceinline__
    value_type operator[](difference_type n) const { return value + static_cast<T>(n); }

    __device__ __forceinline__
    counting_iterator_t& operator++() { ++value; return *this; }

    __device__ __forceinline__
    counting_iterator_t operator++(int) { auto tmp = *this; ++value; return tmp; }

    __device__ __forceinline__
    counting_iterator_t& operator+=(difference_type n) { value += static_cast<T>(n); return *this; }

    __device__ __forceinline__
    counting_iterator_t operator+(difference_type n) const { return counting_iterator_t(value + static_cast<T>(n)); }

    __device__ __forceinline__
    friend counting_iterator_t operator+(difference_type n, const counting_iterator_t& it) {
        return it + n;
    }
};

//==============================================================================
// Constant Iterator
//
// Built-in iterator that always returns the same value
//==============================================================================

template <typename T>
struct constant_iterator_t {
    using value_type = T;
    using difference_type = int64_t;
    using pointer = T*;
    using reference = T;

    T value;
    int64_t index;

    __host__ __device__ constant_iterator_t() : value(T(0)), index(0) {}
    __host__ __device__ explicit constant_iterator_t(T v) : value(v), index(0) {}

    __device__ __forceinline__
    value_type operator*() const { return value; }

    __device__ __forceinline__
    value_type operator[](difference_type) const { return value; }

    __device__ __forceinline__
    constant_iterator_t& operator++() { ++index; return *this; }

    __device__ __forceinline__
    constant_iterator_t& operator+=(difference_type n) { index += n; return *this; }

    __device__ __forceinline__
    constant_iterator_t operator+(difference_type n) const {
        auto tmp = *this;
        tmp.index += n;
        return tmp;
    }

    __device__ __forceinline__
    friend constant_iterator_t operator+(difference_type n, const constant_iterator_t& it) {
        return it + n;
    }
};

#endif // CCCL_JIT_TEMPLATES_ITERATOR_H
