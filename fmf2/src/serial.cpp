#include "fmf2.hpp"

#include <cmath>

#define STABILITY_THRESHOLD 0.000001f

// Calculate the network correlation for a single template against one sample of data
float network_correlation(const float* temp, const float* sum_sq_template,
			  const int* template_moveouts, const float* template_weights,
			  const float* data, const int n_samples_template,
			  const int n_samples_data, const int n_stations, const int n_components,
			  const int normalize);

int matched_filter_serial(const float* templates, const float* sum_square_templates, const int* moveouts, const float* data,
			  const float* weights, const int step, const int n_samples_template,
			  const int n_samples_data, const int n_templates,
			  const int n_stations, const int n_components, const int n_corr,
			  const int normalize, float* cc_sum) {
    const size_t network_size = static_cast<size_t>(n_stations * n_components);

    for (int t = 0; t < n_templates; ++t) {
        const size_t network_offset = t * network_size;
        const size_t cc_sum_offset = t * n_corr;

        int min_moveout = 0;
        int max_moveout = 0;
        for (size_t ch = 0; ch < network_size; ++ch) {
            if (moveouts[network_offset + ch] < min_moveout) min_moveout = moveouts[network_offset + ch];
            if (moveouts[network_offset + ch] > max_moveout) max_moveout = moveouts[network_offset + ch];
        }
        const int start_i = static_cast<int>(std::ceil(std::abs(min_moveout)
						       / static_cast<float>(step)) * step);
        const int stop_i = 1 + (n_samples_data - n_samples_template - max_moveout);

	#pragma omp parallel for schedule(static)
	for (int i = start_i; i < stop_i; i += step) {
	    const auto cc_sum_i_offset = i / step;
	    cc_sum[cc_sum_offset + cc_sum_i_offset] =
		network_correlation(
		    templates + network_offset * n_samples_template,
		    sum_square_templates + network_offset,
		    moveouts + network_offset,
		    weights + network_offset,
		    data + i,
		    n_samples_template,
		    n_samples_data,
		    n_stations,
		    n_components,
		    normalize);
	}
    }
    return 0;
}

float network_correlation(const float* temp, const float* sum_sq_template,
			  const int* template_moveouts, const float* template_weights,
			  const float* data, const int n_samples_template,
			  const int n_samples_data, const int n_stations, const int n_components,
			  const int normalize) {
    float cc_sum = 0.0f;
    for (int station = 0; station < n_stations; station++) {
	for (int comp = 0; comp < n_components; comp++) {
	    const int comp_offset = station * n_components + comp;
	    const int temp_offset = comp_offset * n_samples_template;
	    const int data_offset = comp_offset * n_samples_data + template_moveouts[comp_offset];

	    float mean = 0.0f;
	    float numerator = 0.0f;
	    float sum_square = 0.0f;

	    if (normalize) {
		for (int i = 0; i < n_samples_template; i+=1) {
		    mean += data[data_offset + i];
		}
		mean /= static_cast<float>(n_samples_template);
	    }
	    for (int i = 0; i < n_samples_template; i+=1) {
		const auto sample = data[data_offset + i] - mean;
		const auto templ = temp[temp_offset + i];
		numerator = std::fma(templ, sample, numerator);
		sum_square = std::fma(sample, sample, sum_square);
	    }
	    const float denominator = std::sqrt(sum_sq_template[comp_offset] * sum_square);
	    if (denominator > STABILITY_THRESHOLD) {
		cc_sum = std::fma((numerator / denominator), template_weights[comp_offset], cc_sum);
	    }
	}
    }
    return cc_sum;
}
