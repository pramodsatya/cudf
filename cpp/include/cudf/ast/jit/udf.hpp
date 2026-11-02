/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <functional>
#include <string>
#include <vector>

/**
 * @file
 * @brief Calls of consumer-supplied device functions from JIT expressions.
 */

namespace CUDF_EXPORT cudf {
namespace ast::jit {
/**
 * @addtogroup expressions
 * @{
 */

/**
 * @brief The output or an input of a device function that a JIT expression calls.
 */
struct function_value {
  data_type type;            ///< The value's type
  bool is_nullable = false;  ///< Whether the value is passed as a `cuda::std::optional`

  /**
   * @brief Compares the type and nullability of two values.
   *
   * @param other The value to compare with
   * @return Whether the values are equal
   */
  bool operator==(function_value const& other) const = default;
};

/**
 * @brief A device function that JIT expressions call.
 *
 * The function is defined in one of the LTO fragments passed to the JIT evaluator, such as
 * `cudf::compute_column_jit`, which links all of them into its kernel; `symbol` selects the
 * function. Its parameters follow the conventions of `cudf::lto_fragment`: a pointer to the output,
 * then the inputs by value. It returns `cudf::errc::SUCCESS`, or a code that fails the row, such as
 * `cudf::errc::USER_RESERVED + n`. A function with an `INT32` output and two `INT32` inputs, the
 * second of them nullable, is defined as
 *
 * ```cpp
 * extern "C" __device__ cudf::errc f(int32_t* out, int32_t a, cuda::std::optional<int32_t> b);
 * ```
 *
 * The generated code declares the function from `output` and `inputs`, which must match its
 * definition; a mismatch has undefined behavior.
 *
 * - An input that is not nullable is passed as a value. A row whose argument for it is null skips
 *   the call, and its result is null.
 * - A nullable input is passed as a `cuda::std::optional`, empty for a null argument.
 * - A function with a nullable output writes a `cuda::std::optional`, and leaves it empty to
 *   produce a null.
 * - cuDF assumes the function is pure: calls with equal arguments return equal results, and no
 *   side effect has to happen on every call. Equal calls may then share one evaluation. Set
 *   `is_pure` to `false` for a function that does not meet this, for example one that reads a
 *   clock or a random state, so that each of its calls is evaluated. cuDF cannot check this
 *   setting, and a wrong one changes results.
 * - The function must be row-local: no warp-collective operations, `__syncthreads` or shared
 *   memory.
 */
struct function {
  std::string symbol;                  ///< The function's unmangled name
  bool is_pure = true;                 ///< Whether equal calls may share one evaluation
  function_value output;               ///< The value the function writes
  std::vector<function_value> inputs;  ///< The values the function reads, in parameter order

  /**
   * @brief Compares the name, purity and signature of two functions.
   *
   * @param other The function to compare with
   * @return Whether the functions are equal
   */
  bool operator==(function const& other) const = default;
};

/**
 * @brief Creates a JIT expression that calls a device function.
 *
 * The call is part of the generated JIT kernel, so it fuses with the surrounding expression. It
 * can only be evaluated by the JIT evaluators, which are given the LTO fragments that define the
 * function. All calls of one symbol in the evaluated expressions must declare the same function.
 *
 * A row whose call fails raises `cudf::evaluation_error` under `error_policy::PROPAGATE`, and
 * produces null under `error_policy::NULLIFY`.
 *
 * @throws std::invalid_argument if `callee.output.type` is not fixed-width
 *
 * @param tree The expression tree to which this expression will be added
 * @param callee The device function to call
 * @param args The call's arguments, one for each of `callee.inputs` and of its type
 * @param error_policy How a row whose call fails is handled
 * @return A reference to the created call expression
 */
expression const& call(ast::tree& tree,
                       function const& callee,
                       std::vector<std::reference_wrapper<expression const>> const& args,
                       cudf::error_policy error_policy = cudf::error_policy::PROPAGATE);

/** @} */  // end of group
}  // namespace ast::jit
}  // namespace CUDF_EXPORT cudf
