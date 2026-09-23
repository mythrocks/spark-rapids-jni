/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/operators/operators.cuh>
#include <cudf/errc.hpp>

#include <cuda/std/cstdint>
#include <cuda/std/expected>

#if defined(ADD_OVERFLOW_LTO_INT8)
using add_type = int8_t;
#elif defined(ADD_OVERFLOW_LTO_INT16)
using add_type = int16_t;
#elif defined(ADD_OVERFLOW_LTO_INT32)
using add_type = int32_t;
#elif defined(ADD_OVERFLOW_LTO_INT64)
using add_type = int64_t;
#else
#error "Define one ADD_OVERFLOW_LTO_INT{8,16,32,64} macro"
#endif

extern "C" __device__ int transform(add_type* out, add_type lhs, add_type rhs)
{
  auto result = cudf::detail::ops::add_overflow(lhs, rhs);
  if (!result) { return static_cast<int>(result.error()); }

  *out = result.value();
  return static_cast<int>(cudf::errc::SUCCESS);
}
