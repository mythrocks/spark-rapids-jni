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

#include "add.hpp"
#include "spark_rapids_jni_fragments.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/scalar_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

namespace spark_rapids_jni {
namespace {

bool is_supported_checked_add_type(cudf::data_type type)
{
  switch (type.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64: return true;
    default: return false;
  }
}

void check_add_inputs(cudf::data_type left_type,
                      cudf::data_type right_type,
                      cudf::size_type left_size,
                      cudf::size_type right_size,
                      bool is_ansi_mode,
                      bool is_try_mode)
{
  CUDF_EXPECTS(left_type == right_type, "Input columns must have the same data type");
  CUDF_EXPECTS(left_size == right_size, "Input columns must have the same size");
  CUDF_EXPECTS(!(is_ansi_mode && is_try_mode),
               "Cannot enable both ANSI mode and TRY mode at the same time");
  CUDF_EXPECTS(!is_ansi_mode || is_supported_checked_add_type(left_type),
               "Unsupported data type for ANSI addition.");
  CUDF_EXPECTS(!is_try_mode || is_supported_checked_add_type(left_type),
               "Unsupported data type for TRY addition.");
}

std::span<std::uint8_t const> fragment_for_checked_add(cudf::data_type type)
{
  std::size_t fragment_index;
  switch (type.id()) {
    case cudf::type_id::INT8: fragment_index = spark_rapids_jni_fragments::checked_add_int8; break;
    case cudf::type_id::INT16:
      fragment_index = spark_rapids_jni_fragments::checked_add_int16;
      break;
    case cudf::type_id::INT32:
      fragment_index = spark_rapids_jni_fragments::checked_add_int32;
      break;
    case cudf::type_id::INT64:
      fragment_index = spark_rapids_jni_fragments::checked_add_int64;
      break;
    default: CUDF_FAIL("Unsupported data type for ANSI addition.");
  }

  auto const range = spark_rapids_jni_fragments::file_ranges[fragment_index];
  return spark_rapids_jni_fragments::files.subspan(range[0], range[1]);
}

std::span<std::uint8_t const> fragment_for_try_add(cudf::data_type type)
{
  std::size_t fragment_index;
  switch (type.id()) {
    case cudf::type_id::INT8: fragment_index = spark_rapids_jni_fragments::try_add_int8; break;
    case cudf::type_id::INT16: fragment_index = spark_rapids_jni_fragments::try_add_int16; break;
    case cudf::type_id::INT32: fragment_index = spark_rapids_jni_fragments::try_add_int32; break;
    case cudf::type_id::INT64: fragment_index = spark_rapids_jni_fragments::try_add_int64; break;
    default: CUDF_FAIL("Unsupported data type for TRY addition.");
  }

  auto const range = spark_rapids_jni_fragments::file_ranges[fragment_index];
  return spark_rapids_jni_fragments::files.subspan(range[0], range[1]);
}

std::unique_ptr<cudf::column> checked_add_lto(std::span<cudf::transform_input const> inputs,
                                              cudf::data_type output_type,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  cudf::transform_output outputs[] = {{output_type, cudf::output_nullability::PRESERVE}};
  auto result_table                = cudf::transform_lto(fragment_for_checked_add(output_type),
                                          cudf::lto_binary_type::FATBIN,
                                          cudf::null_aware::NO,
                                          std::nullopt,
                                          inputs,
                                          outputs,
                                                         {},
                                          std::nullopt,
                                          stream,
                                          mr);

  auto result_columns = result_table->release();
  CUDF_EXPECTS(result_columns.size() == 1, "Expected one output column from checked add.");
  return std::move(result_columns[0]);
}

std::unique_ptr<cudf::column> try_add_lto(std::span<cudf::transform_input const> inputs,
                                          cudf::data_type output_type,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  cudf::transform_output outputs[] = {
    {output_type, cudf::output_nullability::PRESERVE},
    {cudf::data_type{cudf::type_id::BOOL8}, cudf::output_nullability::PRESERVE}};
  auto result_table = cudf::transform_lto(fragment_for_try_add(output_type),
                                          cudf::lto_binary_type::FATBIN,
                                          cudf::null_aware::NO,
                                          std::nullopt,
                                          inputs,
                                          outputs,
                                          {},
                                          std::nullopt,
                                          stream,
                                          mr);

  auto result_columns = result_table->release();
  CUDF_EXPECTS(result_columns.size() == 2, "Expected two output columns from try add.");

  auto [null_mask, null_count] = cudf::bools_to_mask(result_columns[1]->view(), stream, mr);
  if (null_count > 0) {
    result_columns[0]->set_null_mask(std::move(*null_mask.release()), null_count);
  }

  return std::move(result_columns[0]);
}

std::unique_ptr<cudf::column> add_lto(std::span<cudf::transform_input const> inputs,
                                      cudf::data_type output_type,
                                      bool is_ansi_mode,
                                      bool is_try_mode,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  if (is_ansi_mode) { return checked_add_lto(inputs, output_type, stream, mr); }
  if (is_try_mode) { return try_add_lto(inputs, output_type, stream, mr); }
  CUDF_FAIL("LTO add requires ANSI or TRY mode.");
}

}  // anonymous namespace

std::unique_ptr<cudf::column> add(cudf::column_view const& left_cv,
                                  cudf::column_view const& right_cv,
                                  bool is_ansi_mode,
                                  bool is_try_mode,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  check_add_inputs(
    left_cv.type(), right_cv.type(), left_cv.size(), right_cv.size(), is_ansi_mode, is_try_mode);

  if (!is_ansi_mode && !is_try_mode) {
    return cudf::binary_operation(
      left_cv, right_cv, cudf::binary_operator::ADD, left_cv.type(), stream, mr);
  }

  cudf::transform_input inputs[] = {left_cv, right_cv};
  return add_lto(inputs, left_cv.type(), is_ansi_mode, is_try_mode, stream, mr);
}

std::unique_ptr<cudf::column> add(cudf::column_view const& left_cv,
                                  cudf::scalar const& right_scalar,
                                  bool is_ansi_mode,
                                  bool is_try_mode,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  check_add_inputs(
    left_cv.type(), right_scalar.type(), left_cv.size(), left_cv.size(), is_ansi_mode, is_try_mode);

  if (!is_ansi_mode && !is_try_mode) {
    return cudf::binary_operation(
      left_cv, right_scalar, cudf::binary_operator::ADD, left_cv.type(), stream, mr);
  }

  auto right_scalar_column       = cudf::make_column_from_scalar(right_scalar, 1, stream, mr);
  auto right_scalar_view         = cudf::scalar_column_view{right_scalar_column->view()};
  cudf::transform_input inputs[] = {left_cv, right_scalar_view};
  return add_lto(inputs, left_cv.type(), is_ansi_mode, is_try_mode, stream, mr);
}

std::unique_ptr<cudf::column> add(cudf::scalar const& left_scalar,
                                  cudf::column_view const& right_cv,
                                  bool is_ansi_mode,
                                  bool is_try_mode,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  check_add_inputs(left_scalar.type(),
                   right_cv.type(),
                   right_cv.size(),
                   right_cv.size(),
                   is_ansi_mode,
                   is_try_mode);

  if (!is_ansi_mode && !is_try_mode) {
    return cudf::binary_operation(
      left_scalar, right_cv, cudf::binary_operator::ADD, right_cv.type(), stream, mr);
  }

  auto left_scalar_column        = cudf::make_column_from_scalar(left_scalar, 1, stream, mr);
  auto left_scalar_view          = cudf::scalar_column_view{left_scalar_column->view()};
  cudf::transform_input inputs[] = {left_scalar_view, right_cv};
  return add_lto(inputs, right_cv.type(), is_ansi_mode, is_try_mode, stream, mr);
}

}  // namespace spark_rapids_jni
