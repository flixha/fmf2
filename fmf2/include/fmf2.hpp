#pragma once

/**
* Header file for the C++ FMF2 source code
*/

int matched_filter_serial(const float* templates, const float* sum_square_templates, const int* moveouts, const float* data,
			  const float* weights, const int step, const int n_samples_template,
			  const int n_samples_data, const int n_templates,
			  const int n_stations, const int n_components, const int n_corr,
			  const int normalize, float* cc_sum);

int matched_filter_sycl(const float* templates, const float* sum_square_templates, const int* moveouts, const float* data,
			const float* weights, const int step, const int n_samples_template,
			const int n_samples_data, const int n_templates,
			const int n_stations, const int n_components, const int n_corr,
			const int normalize, float* cc_sum);
