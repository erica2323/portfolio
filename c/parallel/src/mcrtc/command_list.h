//==============================================================================
// MACA CCCL - Command List Pattern
// Orchestrates JIT compilation and linking (equivalent to NVIDIA's command_list.h)
//==============================================================================

#ifndef CCCL_MCRTC_COMMAND_LIST_H
#define CCCL_MCRTC_COMMAND_LIST_H

#include "mcrtc_helper.h"
#include "mcjitlink_helper.h"
#include <tuple>
#include <string>
#include <vector>
#include <variant>
#include <functional>

namespace cccl {
namespace mcrtc {

//==============================================================================
// Command Data Structures
//==============================================================================

/**
 * Represents compiled code output
 */
struct mcrtc_code {
    std::vector<char> code;

    mcrtc_code() = default;
    explicit mcrtc_code(std::vector<char> c) : code(std::move(c)) {}

    bool empty() const { return code.empty(); }
    size_t size() const { return code.size(); }
    const char* data() const { return code.data(); }
};

/**
 * Represents LTOIR (Link Time Optimization IR) output
 */
struct mcrtc_ltoir {
    std::vector<char> ltoir;
    std::string name;

    mcrtc_ltoir() = default;
    mcrtc_ltoir(std::vector<char> l, std::string n)
        : ltoir(std::move(l)), name(std::move(n)) {}

    bool empty() const { return ltoir.empty(); }
    size_t size() const { return ltoir.size(); }
    const char* data() const { return ltoir.data(); }
};

/**
 * Represents linked binary result
 */
struct mcrtc_link_result {
    std::vector<char> binary;
    std::string log;

    mcrtc_link_result() = default;

    bool empty() const { return binary.empty(); }
    size_t size() const { return binary.size(); }
    const char* data() const { return binary.data(); }
};

/**
 * Represents a translation unit (source code to be compiled)
 */
struct mcrtc_translation_unit {
    std::string source;
    std::string name;
    std::vector<std::string> options;

    mcrtc_translation_unit() = default;
    mcrtc_translation_unit(std::string src, std::string n, std::vector<std::string> opts = {})
        : source(std::move(src)), name(std::move(n)), options(std::move(opts)) {}
};

/**
 * Represents a name expression for kernel name resolution
 */
struct mcrtc_expression {
    std::string expression;

    mcrtc_expression() = default;
    explicit mcrtc_expression(std::string expr) : expression(std::move(expr)) {}
};

/**
 * Compile command with arguments
 */
struct mcrtc_compile {
    std::vector<std::string> args;

    mcrtc_compile() = default;
    explicit mcrtc_compile(std::vector<std::string> a) : args(std::move(a)) {}

    void add_arg(const std::string& arg) { args.push_back(arg); }
    void add_include(const std::string& path) { args.push_back("-I" + path); }
    void add_define(const std::string& def) { args.push_back("-D" + def); }
};

//==============================================================================
// Command List Builder
//==============================================================================

/**
 * Builder for constructing compilation command chains
 */
template <typename... Commands>
class CommandListBuilder {
public:
    std::tuple<Commands...> commands;

    CommandListBuilder() = default;
    explicit CommandListBuilder(std::tuple<Commands...> cmds)
        : commands(std::move(cmds)) {}

    /**
     * Add a new command to the chain
     */
    template <typename NewCommand>
    auto add(NewCommand cmd) const {
        auto new_commands = std::tuple_cat(commands, std::make_tuple(std::move(cmd)));
        return CommandListBuilder<Commands..., NewCommand>(std::move(new_commands));
    }

    /**
     * Add a translation unit
     */
    auto add_source(std::string source, std::string name) const {
        return add(mcrtc_translation_unit(std::move(source), std::move(name)));
    }

    /**
     * Add compile arguments
     */
    auto add_compile_args(std::vector<std::string> args) const {
        return add(mcrtc_compile(std::move(args)));
    }

    /**
     * Add a name expression
     */
    auto add_expression(std::string expr) const {
        return add(mcrtc_expression(std::move(expr)));
    }
};

/**
 * Create an empty command list builder
 */
inline CommandListBuilder<> make_command_list() {
    return CommandListBuilder<>();
}

//==============================================================================
// Compilation Unit Builder
//==============================================================================

/**
 * Builds a single compilation unit with source, options, and expressions
 */
class CompilationUnit {
public:
    CompilationUnit() = default;

    CompilationUnit& set_source(std::string source, std::string name) {
        source_ = std::move(source);
        name_ = std::move(name);
        return *this;
    }

    CompilationUnit& add_option(const std::string& opt) {
        options_.push_back(opt);
        return *this;
    }

    CompilationUnit& add_include(const std::string& path) {
        options_.push_back("-I" + path);
        return *this;
    }

    CompilationUnit& add_define(const std::string& name, const std::string& value = "") {
        if (value.empty()) {
            options_.push_back("-D" + name);
        } else {
            options_.push_back("-D" + name + "=" + value);
        }
        return *this;
    }

    CompilationUnit& add_expression(const std::string& expr) {
        expressions_.push_back(expr);
        return *this;
    }

    /**
     * Compile to LTOIR for linking
     */
    CompileResult compile_to_ltoir() const {
        auto opts = options_;
        opts.push_back("-dlto");  // Enable LTO

        return Compiler::compileToLTOIR(source_, name_, opts);
    }

    /**
     * Compile to final binary
     */
    CompileResult compile() const {
        return Compiler::compile(source_, name_, options_, expressions_);
    }

    const std::string& source() const { return source_; }
    const std::string& name() const { return name_; }
    const std::vector<std::string>& options() const { return options_; }
    const std::vector<std::string>& expressions() const { return expressions_; }

private:
    std::string source_;
    std::string name_;
    std::vector<std::string> options_;
    std::vector<std::string> expressions_;
};

//==============================================================================
// Top-Level JIT Compilation Manager
//==============================================================================

/**
 * Manages the full JIT compilation workflow
 */
class JitCompilationManager {
public:
    struct BuildConfig {
        std::string arch;              // Target architecture (e.g., "gfx906")
        std::vector<std::string> include_paths;
        std::vector<std::string> defines;
        bool enable_lto = true;
        bool debug_info = false;
        int opt_level = 2;             // -O2
    };

    JitCompilationManager() = default;
    explicit JitCompilationManager(BuildConfig config) : config_(std::move(config)) {}

    /**
     * Add a compilation unit
     */
    void add_unit(CompilationUnit unit) {
        units_.push_back(std::move(unit));
    }

    /**
     * Add user-provided LTOIR (pre-compiled operation)
     */
    void add_user_ltoir(const void* data, size_t size, const std::string& name) {
        mcrtc_ltoir ltoir;
        ltoir.ltoir.assign(static_cast<const char*>(data),
                          static_cast<const char*>(data) + size);
        ltoir.name = name;
        user_ltoirs_.push_back(std::move(ltoir));
    }

    /**
     * Compile all units and link into final binary
     */
    LinkResult build() {
        LinkResult result;

        // Build common options
        std::vector<std::string> common_opts = build_options();

        // Compile all units to LTOIR
        std::vector<Linker::LTOIRInput> link_inputs;

        for (auto& unit : units_) {
            // Add common options to unit
            for (const auto& opt : common_opts) {
                unit.add_option(opt);
            }

            auto compile_result = unit.compile_to_ltoir();
            if (!compile_result.success) {
                result.error_log = "Compilation failed for " + unit.name() + ": " +
                                   compile_result.log;
                return result;
            }

            // Store LTOIR for linking
            compiled_ltoirs_.push_back(mcrtc_ltoir(
                std::move(compile_result.ltoir),
                unit.name()
            ));
        }

        // Add compiled LTOIRs to link inputs
        for (const auto& ltoir : compiled_ltoirs_) {
            if (!ltoir.empty()) {
                link_inputs.push_back({ltoir.data(), ltoir.size(), ltoir.name});
            }
        }

        // Add user-provided LTOIRs
        for (const auto& ltoir : user_ltoirs_) {
            if (!ltoir.empty()) {
                link_inputs.push_back({ltoir.data(), ltoir.size(), ltoir.name});
            }
        }

        // Link all inputs
        std::vector<std::string> link_opts;
        if (!config_.arch.empty()) {
            link_opts.push_back("-arch=" + config_.arch);
        }

        return Linker::link(link_inputs, link_opts);
    }

    /**
     * Clear all compilation units and cached results
     */
    void clear() {
        units_.clear();
        compiled_ltoirs_.clear();
        user_ltoirs_.clear();
    }

private:
    std::vector<std::string> build_options() const {
        std::vector<std::string> opts;

        // Standard
        opts.push_back("-std=c++17");

        // Architecture
        if (!config_.arch.empty()) {
            opts.push_back("-arch=" + config_.arch);
        }

        // Optimization
        opts.push_back("-O" + std::to_string(config_.opt_level));

        // Include paths
        for (const auto& path : config_.include_paths) {
            opts.push_back("-I" + path);
        }

        // Defines
        for (const auto& def : config_.defines) {
            opts.push_back("-D" + def);
        }

        // Debug info
        if (config_.debug_info) {
            opts.push_back("-g");
            opts.push_back("-lineinfo");
        }

        return opts;
    }

    BuildConfig config_;
    std::vector<CompilationUnit> units_;
    std::vector<mcrtc_ltoir> compiled_ltoirs_;
    std::vector<mcrtc_ltoir> user_ltoirs_;
};

} // namespace mcrtc
} // namespace cccl

#endif // CCCL_MCRTC_COMMAND_LIST_H
