//*****************************************************************************
// Copyright (c) 2016-2020, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

/*
 * This header file contains single argument element wise functions definitions
 *
 * Macro `MACRO_CUSTOM_1ARG_2TYPES_OP` must be defined before usage
 *
 * Parameters:
 * - public name of the function and kernel name
 * - operation used to calculate the result
 *
 */

#ifndef MACRO_CUSTOM_1ARG_2TYPES_OP
#error "MACRO_CUSTOM_1ARG_2TYPES_OP is not defined"
#endif

MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_acos_c, cl::sycl::acos(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_acosh_c, cl::sycl::acosh(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_asin_c, cl::sycl::asin(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_asinh_c, cl::sycl::asinh(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_atan_c, cl::sycl::atan(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_atanh_c, cl::sycl::atanh(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_cbrt_c, cl::sycl::cbrt(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_ceil_c, cl::sycl::ceil(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_cos_c, cl::sycl::cos(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_cosh_c, cl::sycl::cosh(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_degrees_c, cl::sycl::degrees(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_exp2_c, cl::sycl::exp2(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_exp_c, cl::sycl::exp(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_expm1_c, cl::sycl::expm1(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_fabs_c, cl::sycl::fabs(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_floor_c, cl::sycl::floor(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_log10_c, cl::sycl::log10(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_log1p_c, cl::sycl::log1p(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_log2_c, cl::sycl::log2(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_log_c, cl::sycl::log(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_radians_c, cl::sycl::radians(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_sin_c, cl::sycl::sin(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_sinh_c, cl::sycl::sinh(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_sqrt_c, cl::sycl::sqrt(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_tan_c, cl::sycl::tan(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_tanh_c, cl::sycl::tanh(input_elem))
MACRO_CUSTOM_1ARG_2TYPES_OP(dpnp_trunc_c, cl::sycl::trunc(input_elem))

#undef MACRO_CUSTOM_1ARG_2TYPES_OP
