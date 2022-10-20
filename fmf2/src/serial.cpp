#include "fmf2.hpp"

#include <cmath>
#include <immintrin.h>

#define STABILITY_THRESHOLD 0.000001f

// Calculate the network correlation for a single template against one sample of
// data
float network_correlation(const float *temp, const float *sum_sq_template,
                          const int *template_moveouts,
                          const float *template_weights, const float *data,
                          const int n_samples_template,
                          const int n_samples_data, const int n_stations,
                          const int n_components, const int normalize);

// Manually optimized AVX2 version of the above
float network_correlation_avx2(const float *temp, const float *sum_sq_template,
                               const int *template_moveouts,
                               const float *template_weights, const float *data,
                               const int n_samples_template,
                               const int n_samples_data, const int n_stations,
                               const int n_components, const int normalize);

float network_correlation_avx512(
    const float *temp, const float *sum_sq_template,
    const int *template_moveouts, const float *template_weights,
    const float *data, const int n_samples_template, const int n_samples_data,
    const int n_stations, const int n_components, const int normalize);

int matched_filter_serial(
    const float *templates, const float *sum_square_templates,
    const int *moveouts, const float *data, const float *weights,
    const int step, const int n_samples_template, const int n_samples_data,
    const int n_templates, const int n_stations, const int n_components,
    const int n_corr, const int normalize, float *cc_sum) {
  const size_t network_size = static_cast<size_t>(n_stations * n_components);

  for (int t = 0; t < n_templates; ++t) {
    const size_t network_offset = t * network_size;
    const size_t cc_sum_offset = t * n_corr;

    int min_moveout = 0;
    int max_moveout = 0;
    for (size_t ch = 0; ch < network_size; ++ch) {
      if (moveouts[network_offset + ch] < min_moveout)
        min_moveout = moveouts[network_offset + ch];
      if (moveouts[network_offset + ch] > max_moveout)
        max_moveout = moveouts[network_offset + ch];
    }
    const int start_i = static_cast<int>(
        std::ceil(std::abs(min_moveout) / static_cast<float>(step)) * step);
    const int stop_i = 1 + (n_samples_data - n_samples_template - max_moveout);

#pragma omp parallel for schedule(static)
    for (int i = start_i; i < stop_i; i += step) {
      const auto cc_sum_i_offset = i / step;
#ifdef __AVX512F__
#pragma message "Using AVX512 CPU path"
      cc_sum[cc_sum_offset + cc_sum_i_offset] = network_correlation_avx512(
          templates + network_offset * n_samples_template,
          sum_square_templates + network_offset, moveouts + network_offset,
          weights + network_offset, data + i, n_samples_template,
          n_samples_data, n_stations, n_components, normalize);
#elif __AVX2__ && __FMA__
#pragma message "Using AVX2 CPU path"
      cc_sum[cc_sum_offset + cc_sum_i_offset] = network_correlation_avx2(
          templates + network_offset * n_samples_template,
          sum_square_templates + network_offset, moveouts + network_offset,
          weights + network_offset, data + i, n_samples_template,
          n_samples_data, n_stations, n_components, normalize);
#else
      cc_sum[cc_sum_offset + cc_sum_i_offset] = network_correlation(
          templates + network_offset * n_samples_template,
          sum_square_templates + network_offset, moveouts + network_offset,
          weights + network_offset, data + i, n_samples_template,
          n_samples_data, n_stations, n_components, normalize);
#endif
    }
  }
  return 0;
}

float network_correlation(const float *temp, const float *sum_sq_template,
                          const int *template_moveouts,
                          const float *template_weights, const float *data,
                          const int n_samples_template,
                          const int n_samples_data, const int n_stations,
                          const int n_components, const int normalize) {
  float cc_sum = 0.0f;
  for (int station = 0; station < n_stations; station++) {
    for (int comp = 0; comp < n_components; comp++) {
      const int comp_offset = station * n_components + comp;
      const int temp_offset = comp_offset * n_samples_template;
      const int data_offset =
          comp_offset * n_samples_data + template_moveouts[comp_offset];

      float mean = 0.0f;
      float numerator = 0.0f;
      float sum_square = 0.0f;

      if (normalize) {
        for (int i = 0; i < n_samples_template; i += 1) {
          mean += data[data_offset + i];
        }
        mean /= static_cast<float>(n_samples_template);
      }
      for (int i = 0; i < n_samples_template; i += 1) {
        const auto sample = data[data_offset + i] - mean;
        const auto templ = temp[temp_offset + i];
        numerator = std::fma(templ, sample, numerator);
        sum_square = std::fma(sample, sample, sum_square);
      }
      const float denominator =
          std::sqrt(sum_sq_template[comp_offset] * sum_square);
      if (denominator > STABILITY_THRESHOLD) {
        cc_sum = std::fma((numerator / denominator),
                          template_weights[comp_offset], cc_sum);
      }
    }
  }
  return cc_sum;
}

// Horizontal sum of a vector register
// https://stackoverflow.com/a/13222410
#ifdef __AVX2__
static float sum8(__m256 x) {
  // hiQuad = ( x7, x6, x5, x4 )
  const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
  // loQuad = ( x3, x2, x1, x0 )
  const __m128 loQuad = _mm256_castps256_ps128(x);
  // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
  const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
  // loDual = ( -, -, x1 + x5, x0 + x4 )
  const __m128 loDual = sumQuad;
  // hiDual = ( -, -, x3 + x7, x2 + x6 )
  const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
  // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
  const __m128 sumDual = _mm_add_ps(loDual, hiDual);
  // lo = ( -, -, -, x0 + x2 + x4 + x6 )
  const __m128 lo = sumDual;
  // hi = ( -, -, -, x1 + x3 + x5 + x7 )
  const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
  const __m128 sum = _mm_add_ss(lo, hi);
  return _mm_cvtss_f32(sum);
}
#endif

float network_correlation_avx2(const float *temp, const float *sum_sq_template,
                               const int *template_moveouts,
                               const float *template_weights, const float *data,
                               const int n_samples_template,
                               const int n_samples_data, const int n_stations,
                               const int n_components, const int normalize) {
  float cc_sum = 0.0f;
#ifdef __AVX2__
  const auto left = n_samples_template % 8;
  const int offset = n_samples_template - left;
  const __m256i mask =
      _mm256_set_epi32((offset + 0 < n_samples_template) ? 0 : -1,
                       (offset + 1 < n_samples_template) ? 0 : -1,
                       (offset + 2 < n_samples_template) ? 0 : -1,
                       (offset + 3 < n_samples_template) ? 0 : -1,
                       (offset + 4 < n_samples_template) ? 0 : -1,
                       (offset + 5 < n_samples_template) ? 0 : -1,
                       (offset + 6 < n_samples_template) ? 0 : -1,
                       (offset + 7 < n_samples_template) ? 0 : -1);
  for (int station = 0; station < n_stations; station++) {
    for (int comp = 0; comp < n_components; comp++) {
      const int comp_offset = station * n_components + comp;
      const int temp_offset = comp_offset * n_samples_template;
      const int data_offset =
          comp_offset * n_samples_data + template_moveouts[comp_offset];

      __m256 mean = _mm256_setzero_ps();
      __m256 numerator = _mm256_setzero_ps();
      __m256 sum_square = _mm256_setzero_ps();
      float mean_sum = 0;

      if (normalize) {
        for (int i = 0; (i + 8) < n_samples_template; i += 8) {
          mean += _mm256_loadu_ps(&data[data_offset + i]);
        }
        if (left != 0) {
          mean += _mm256_maskload_ps(&data[data_offset + offset], mask);
        }
        mean_sum = sum8(mean) / static_cast<float>(n_samples_template);
        mean = _mm256_set1_ps(mean_sum);
      }
      for (int i = 0; (i + 8) < n_samples_template; i += 8) {
        const __m256 sample = _mm256_loadu_ps(&data[data_offset + i]) - mean;
        const __m256 templ = _mm256_loadu_ps(&temp[temp_offset + i]);
        numerator = _mm256_fmadd_ps(templ, sample, numerator);
        sum_square = _mm256_fmadd_ps(sample, sample, sum_square);
      }
      if (left != 0) {
        mean =
            _mm256_set_ps((offset + 0 < n_samples_template) ? 0.f : mean_sum,
                          (offset + 1 < n_samples_template) ? 0.f : mean_sum,
                          (offset + 2 < n_samples_template) ? 0.f : mean_sum,
                          (offset + 3 < n_samples_template) ? 0.f : mean_sum,
                          (offset + 4 < n_samples_template) ? 0.f : mean_sum,
                          (offset + 5 < n_samples_template) ? 0.f : mean_sum,
                          (offset + 6 < n_samples_template) ? 0.f : mean_sum,
                          (offset + 7 < n_samples_template) ? 0.f : mean_sum);
        const __m256 sample =
            _mm256_maskload_ps(&data[data_offset + offset], mask) - mean;
        const __m256 templ =
            _mm256_maskload_ps(&temp[temp_offset + offset], mask);
        numerator = _mm256_fmadd_ps(templ, sample, numerator);
        sum_square = _mm256_fmadd_ps(sample, sample, sum_square);
      }
      const float denominator =
          std::sqrt(sum_sq_template[comp_offset] * sum8(sum_square));
      if (denominator > STABILITY_THRESHOLD) {
        const float numer = sum8(numerator);
        cc_sum = std::fma((numer / denominator), template_weights[comp_offset],
                          cc_sum);
      }
    }
  }
#endif // __AVX2__
  return cc_sum;
}

float network_correlation_avx512(
    const float *temp, const float *sum_sq_template,
    const int *template_moveouts, const float *template_weights,
    const float *data, const int n_samples_template, const int n_samples_data,
    const int n_stations, const int n_components, const int normalize) {
  float cc_sum = 0.0f;
#ifdef __AVX512F__
  const auto left = n_samples_template % 16;
  const int offset = n_samples_template - left;
  __mmask16 mask = 0;
  for (size_t i = 0; i < 16; i++) {
    mask |= (offset + i < n_samples_template) ? (1 << i) : 0;
  }
  for (int station = 0; station < n_stations; station++) {
    for (int comp = 0; comp < n_components; comp++) {
      const int comp_offset = station * n_components + comp;
      const int temp_offset = comp_offset * n_samples_template;
      const int data_offset =
          comp_offset * n_samples_data + template_moveouts[comp_offset];

      __m512 mean = _mm512_setzero_ps();
      __m512 numerator = _mm512_setzero_ps();
      __m512 sum_square = _mm512_setzero_ps();
      float mean_sum = 0;

      if (normalize) {
        for (int i = 0; (i + 16) < n_samples_template; i += 16) {
          mean += _mm512_loadu_ps(&data[data_offset + i]);
        }
        if (left != 0) {
          mean += _mm512_maskz_loadu_ps(mask, &data[data_offset + offset]);
        }
        mean_sum =
            _mm512_reduce_add_ps(mean) / static_cast<float>(n_samples_template);
        mean = _mm512_set1_ps(mean_sum);
      }
      for (int i = 0; (i + 16) < n_samples_template; i += 16) {
        const __m512 sample = _mm512_loadu_ps(&data[data_offset + i]) - mean;
        const __m512 templ = _mm512_loadu_ps(&temp[temp_offset + i]);
        numerator = _mm512_fmadd_ps(templ, sample, numerator);
        sum_square = _mm512_fmadd_ps(sample, sample, sum_square);
      }
      if (left != 0) {
        const __m512 sample = _mm512_maskz_sub_ps(
            mask, _mm512_maskz_loadu_ps(mask, &data[data_offset + offset]),
            mean);
        const __m512 templ =
            _mm512_maskz_loadu_ps(mask, &temp[temp_offset + offset]);
        numerator = _mm512_fmadd_ps(templ, sample, numerator);
        sum_square = _mm512_fmadd_ps(sample, sample, sum_square);
      }
      const float denominator = std::sqrt(sum_sq_template[comp_offset] *
                                          _mm512_reduce_add_ps(sum_square));
      if (denominator > STABILITY_THRESHOLD) {
        const float numer = _mm512_reduce_add_ps(numerator);
        cc_sum = std::fma((numer / denominator), template_weights[comp_offset],
                          cc_sum);
      }
    }
  }
#endif // __AVX512F__
  return cc_sum;
}
