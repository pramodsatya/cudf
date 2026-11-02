/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A second fragment of device functions that AST JIT expressions call, linked together with
// ast_udf_callees.cu in tests/ast/jit_expressions_tests.cpp.

#include <cudf/errc.hpp>

#include <cstdint>

extern "C" __device__ cudf::errc lto_add_ten(int32_t* out, int32_t a)
{
  *out = a + 10;
  return cudf::errc::SUCCESS;
}
