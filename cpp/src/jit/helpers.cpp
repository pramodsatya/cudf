/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "helpers.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>

#include <jit/cache.hpp>

#include <format>

namespace cudf {
namespace jit {

namespace {
// Compute capability of the current device as a two-digit int (e.g. 90 for
// sm_90). Uses the runtime API so it works on any thread that can see the
// device, without requiring a current driver context.
int current_device_compute_capability()
{
  int device   = 0;
  int cc_major = 0;
  int cc_minor = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device));
  return cc_major * 10 + cc_minor;
}
}  // namespace

bool is_scalar(cudf::size_type base_column_size, cudf::size_type column_size)
{
  return column_size == 1 && column_size != base_column_size;
}

typename std::vector<column_view>::const_iterator get_transform_base_column(
  std::vector<column_view> const& inputs)
{
  if (inputs.empty()) { return inputs.end(); }

  auto [smallest, largest] = std::minmax_element(
    inputs.begin(), inputs.end(), [](auto const& a, auto const& b) { return a.size() < b.size(); });

  /// when the largest size is 1, the size-1 column could be a scalar or an actual column, it would
  /// be a scalar if it has columns that are zero-sized
  if (largest->size() != 1) { return largest; }

  if (smallest->size() == 0) { return smallest; }

  return largest;
}

size_type get_projection_size(column_view const& col) { return col.size(); }

/// @brief Scalar columns don't contribute to the row-size of a transform.
size_type get_projection_size(scalar_column_view const& col) { return 0; }

size_type get_projection_size(std::span<std::variant<column_view, scalar_column_view> const> inputs)
{
  CUDF_EXPECTS(
    !inputs.empty(), "Transform must have at least 1 input column", std::invalid_argument);

  auto get_size = [](auto const& var) {
    return std::visit([](auto& a) { return get_projection_size(a); }, var);
  };

  return *std::max_element(thrust::make_transform_iterator(inputs.begin(), get_size),
                           thrust::make_transform_iterator(inputs.end(), get_size));
}

std::map<uint32_t, std::string> build_ptx_params(std::span<std::string const> output_typenames,
                                                 std::span<std::string const> input_typenames,
                                                 bool has_user_data)
{
  std::map<uint32_t, std::string> params;
  uint32_t index = 0;

  if (has_user_data) {
    params.emplace(index++, "void *");
    params.emplace(index++, jitify2::reflection::reflect<cudf::size_type>());
  }

  for (auto& name : output_typenames) {
    params.emplace(index++, name + "*");
  }

  for (auto& name : input_typenames) {
    params.emplace(index++, name);
  }

  return params;
}

std::vector<std::string> input_type_names(
  std::span<std::variant<column_view, scalar_column_view> const> views)
{
  std::vector<std::string> names;

  std::transform(views.begin(), views.end(), std::back_inserter(names), [&](auto const& view) {
    return std::visit([](auto& a) { return type_to_name(a.type()); }, view);
  });

  return names;
}

jitify2::Kernel get_udf_kernel(jitify2::PreprocessedProgramData const& preprocessed_program_data,
                               std::string const& kernel_name,
                               std::string const& cuda_source,
                               std::vector<std::string> const& extra_options)
{
  CUDF_FUNC_RANGE();

  int runtime_version;
  CUDF_CUDA_TRY(cudaRuntimeGetVersion(&runtime_version));

  constexpr int min_pch_cuda_version     = 12800;  // CUDA 12.8
  constexpr int min_minimal_cuda_version = 12800;  // CUDA 12.8

  std::vector<std::string> options;
  // Pass the concrete device architecture instead of jitify's "-arch=sm_."
  // auto-detect token: the latter resolves the arch via the driver API
  // (cuCtxGetDevice), which fails on a thread that has no current context.
  options.emplace_back(std::format("-arch=sm_{}", current_device_compute_capability()));

  if (runtime_version >= min_minimal_cuda_version) { options.emplace_back("-minimal"); }

  if (runtime_version >= min_pch_cuda_version) { options.emplace_back("-pch"); }

  for (auto& opt : extra_options) {
    options.push_back(opt);
  }

  return cudf::jit::get_program_cache(preprocessed_program_data)
    .get_kernel(kernel_name, {}, {{"cudf/detail/operation-udf.hpp", cuda_source}}, options);
}

}  // namespace jit
}  // namespace cudf
