/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>

namespace spark_rapids_jni {

/**
 * @brief Add two columns with overflow handling if ANSI or try mode is enabled.
 *
 * Regular mode delegates to cudf binary ADD. ANSI and try modes use embedded
 * LTO fragments backed by cudf::detail::ops::add_overflow for Spark integral
 * types.
 *
 * @param left_input The left input.
 * @param right_input The right input.
 * @param is_ansi_mode If true, throws exception when overflow occurs.
 * @param is_try_mode If true, set null when overflow occurs.
 * @param stream CUDA stream to use for the operation.
 * @param mr Memory resource to use for allocations.
 * @return A new column containing the results of the addition.
 */
std::unique_ptr<cudf::column> add(
  cudf::column_view const& left_input,
  cudf::column_view const& right_input,
  bool is_ansi_mode,
  bool is_try_mode,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Add a column and a scalar with overflow handling if ANSI or try mode
 * is enabled.
 *
 * @param left_input The left input.
 * @param right_input The right input.
 * @param is_ansi_mode If true, throws exception when overflow occurs.
 * @param is_try_mode If true, set null when overflow occurs.
 * @param stream CUDA stream to use for the operation.
 * @param mr Memory resource to use for allocations.
 * @return A new column containing the results of the addition.
 */
std::unique_ptr<cudf::column> add(
  cudf::column_view const& left_input,
  cudf::scalar const& right_input,
  bool is_ansi_mode,
  bool is_try_mode,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Add a scalar and a column with overflow handling if ANSI or try mode
 * is enabled.
 *
 * @param left_input The left input.
 * @param right_input The right input.
 * @param is_ansi_mode If true, throws exception when overflow occurs.
 * @param is_try_mode If true, set null when overflow occurs.
 * @param stream CUDA stream to use for the operation.
 * @param mr Memory resource to use for allocations.
 * @return A new column containing the results of the addition.
 */
std::unique_ptr<cudf::column> add(
  cudf::scalar const& left_input,
  cudf::column_view const& right_input,
  bool is_ansi_mode,
  bool is_try_mode,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
