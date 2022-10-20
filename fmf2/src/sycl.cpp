/**
 * SYCL backend for the fast matched filter
 */

// The following pragmas ignore warnings for things included in headers
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <SYCL/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#pragma clang diagnostic pop

#include "fmf2.hpp"

#define STABILITY_THRESHOLD 0.000001f

void println(const std::string &txt) {} // std::cout << txt << std::endl; }

// Helper struct which holds a collection of device pointers, which makes it
// easier to store a group of pointers in vectors
struct DevicePtrs {
  float *templates;
  float *sum_sq_templates;
  float *data;
  float *weights;
  int *moveouts;
  float *cc_sum;
};

std::vector<sycl::device>
enumerate_devices(const int n_samples_template, const int n_samples_data,
                  const int n_templates, const int n_stations,
                  const int n_components, const int n_corr) {
  std::vector<sycl::device> devices;
  // Estimate roughly the amount of device memory we need and filter out
  // devices with too little memory
  const size_t network_size = n_components * n_stations;
  const size_t total_memory =
      sizeof(float) *
      (network_size * n_samples_template + // Storage for 'templates'
       network_size +                      // Storage for 'sum_square_templates'
       network_size * n_samples_data +     // Storage for 'data'
       network_size +                      // Storage for 'weights'
       network_size +                      // Storage for 'moveouts'
       n_corr * n_templates);              // Approximate storage for 'cc_sum'
  // Required size for local memory arrays
  const size_t local_memory =
      sizeof(float) * network_size * 2 + sizeof(int) * network_size +
      sizeof(float) * n_samples_template // Size of templates local data
      +
      sizeof(float) * (n_samples_template + 1024); // Minimum size of data array
  println("------------------------------");
  println("Performing device discovery:");
  println("\tRequire device with at least:\n\t\t" +
          std::to_string(total_memory) + " bytes of global memory\n\t\t" +
          std::to_string(local_memory) + " bytes of local memory");
  for (auto device : sycl::device::get_devices(sycl::info::device_type::gpu)) {
    const auto dev_mem = device.get_info<sycl::info::device::global_mem_size>();
    const auto dev_name = device.get_info<sycl::info::device::name>();
    const auto dev_local_mem =
        device.get_info<sycl::info::device::local_mem_size>();
    if (dev_mem >= total_memory) {
      if (dev_local_mem >= local_memory) {
        println("\tUsing device: " + dev_name);
        devices.push_back(device);
      } else {
        std::cerr << "\033[0;33mWARN: " << dev_name
                  << " has too little local memory, found " << dev_local_mem
                  << " require " << local_memory << " bytes \033[0m\n";
      }
    } else {
      std::cerr << "\033[0;33mWARN: " << dev_name
                << " has too little global memory, found " << dev_mem
                << " require " << total_memory << " bytes \033[0m\n";
    }
  }
  return devices;
}

int matched_filter_sycl(const float *templates,
                        const float *sum_square_templates, const int *moveouts,
                        const float *data, const float *weights, const int step,
                        const int n_samples_template, const int n_samples_data,
                        const int n_templates, const int n_stations,
                        const int n_components, const int n_corr,
                        const int normalize, float *cc_sum) {
  const auto devices =
      enumerate_devices(n_samples_template, n_samples_data, n_templates,
                        n_stations, n_components, n_corr);
  std::vector<sycl::queue> queues;
  if (devices.size() > 0) {
    // If we have more devices than templates we limit the number of
    // devices to reduce the initialization cost
    for (int i = 0; i < std::min(static_cast<int>(devices.size()), n_templates);
         i++) {
      const auto device = devices[i];
      queues.push_back(sycl::queue{device, sycl::property::queue::in_order()});
    }
  } else {
    queues.push_back(sycl::queue{sycl::default_selector{}});
    std::cerr << "\033[0;33mWARN: Did not find any SYCL enabled GPUs, using "
                 "default device\033[0m\n";
  }

  // Allocate device memory per queue/device
  const size_t network_size = static_cast<size_t>(n_stations * n_components);
  std::vector<struct DevicePtrs> device_data;
  // Create one output buffer per queue/device so that each loop iteration
  // below can be run independent
  for (auto Q : queues) {
    // Allocate memory per queue/device
    device_data.push_back({
        sycl::malloc_device<float>(network_size * n_samples_template, Q),
        sycl::malloc_device<float>(network_size, Q),
        sycl::malloc_device<float>(network_size * n_samples_data, Q),
        sycl::malloc_device<float>(network_size, Q),
        sycl::malloc_device<int>(network_size, Q),
        sycl::malloc_device<float>(static_cast<size_t>(n_corr * n_templates),
                                   Q),
    });
    // 'data' never changes so we upload it right away
    auto queue_mem = device_data[device_data.size() - 1];
    // Check that allocated memory got allocated
    if (queue_mem.templates == nullptr ||
        queue_mem.sum_sq_templates == nullptr || queue_mem.data == nullptr ||
        queue_mem.weights == nullptr || queue_mem.moveouts == nullptr ||
        queue_mem.cc_sum == nullptr) {
      const auto device = Q.get_device();
      std::cerr << "\033[0;31mCRITICAL: "
                << "Could not allocate memory on device " << device_data.size()
                << ", (device: " << device.get_info<sycl::info::device::name>()
                << " from " << device.get_info<sycl::info::device::vendor>()
                << " with total memory "
                << device.get_info<sycl::info::device::global_mem_size>()
                << ")\nProgram will exit!\033[0m\n";
      return -28;
    }
    // Copy static data to device (note this never changes)
    Q.copy<float>(data, queue_mem.data, network_size * n_samples_data);
  }

  for (int t = 0; t < n_templates; ++t) {
    const auto queue_index = t % queues.size();
    auto Q = queues[queue_index];
    auto device_mem = device_data[queue_index];

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
    const sycl::range i_range{static_cast<size_t>(stop_i - start_i) / step};

    // Copy iteration data to our current queue
    Q.copy<float>(templates + network_offset * n_samples_template,
                  device_mem.templates, network_size * n_samples_template);
    Q.copy<float>(sum_square_templates + network_offset,
                  device_mem.sum_sq_templates, network_size);
    Q.copy<float>(weights + network_offset, device_mem.weights, network_size);
    Q.copy<int>(moveouts + network_offset, device_mem.moveouts, network_size);
    // To allow SYCL to access local shared memory we run an ND-kernel
    // below, that requires us to define both the global and local size
    // (local size is something akin to blocks in CUDA parlance), the
    // following value has been optimized for large HPC GPUs, one can try
    // to reduce this and see if performance improves on your current
    // device if performance is lower than expected
    const size_t local_size =
        1024; // Fixed work-group size, ensure it is a multiple of 32
    // Since we need the global range to be a multiple of the work-group
    // size we adjust the global work group size here based on the actual
    // ranges calculated above
    size_t global_size = i_range.size();
    if (global_size % local_size != 0) {
      global_size += local_size - (global_size % local_size);
    }
    // First calculate network correlation with all data residing on device
    // this will ensure that all tasks enqueued are run asynchronously and
    // that tasks on different queues can be run in parallel
    Q.submit([&](sycl::handler &h) {
      auto local_sum_sq = sycl::local_accessor<float, 1>(network_size, h);
      auto local_weights = sycl::local_accessor<float, 1>(network_size, h);
      auto local_moveouts = sycl::local_accessor<int, 1>(network_size, h);

      auto local_templ = sycl::local_accessor<float, 1>(n_samples_template, h);
      auto local_data =
          sycl::local_accessor<float, 1>(n_samples_template + local_size, h);

      h.parallel_for(
          sycl::nd_range{sycl::range{global_size}, sycl::range{local_size}},
          [=](auto &idx) {
            const int index = idx.get_global_id()[0] * step + start_i;
            // Move data into local memory as close to cores as possible,
            // this optimization enables us to read from global
            // arrays coalesced and then read from local memory
            // uncoalesced
            const size_t local_index = idx.get_local_id()[0];
            const size_t local_size = idx.get_local_range()[0];
            for (size_t i = local_index; i < network_size; i += local_size) {
              local_sum_sq[i] = device_mem.sum_sq_templates[i];
              local_weights[i] = device_mem.weights[i];
              local_moveouts[i] = device_mem.moveouts[i];
            }
            // Wait for all work-items to update shared local memory
            idx.barrier(sycl::access::fence_space::local_space);

            float cc_sum = 0.0f;
            for (int station = 0; station < n_stations; station++) {
              for (int comp = 0; comp < n_components; comp++) {
                const int comp_offset = station * n_components + comp;
                const int temp_offset = comp_offset * n_samples_template;
                const int data_offset =
                    comp_offset * n_samples_data + local_moveouts[comp_offset];

                // We need to synchronize with the previous iteration
                // so that no in-use local memory is overwritten
                idx.barrier(sycl::access::fence_space::local_space);
                for (int i = local_index; i < n_samples_template;
                     i += local_size) {
                  local_templ[i] = device_mem.templates[temp_offset + i];
                }
                for (int i = 0;
                     i + local_index < n_samples_template + local_size;
                     i += local_size) {
                  local_data[local_index + i] =
                      device_mem.data[index + data_offset + i];
                }
                idx.barrier(sycl::access::fence_space::local_space);

                float mean = 0.0f;
                sycl::float2 numerator{0.0f, 0.0f};
                sycl::float2 sum_square{0.0f, 0.0f};

                if (normalize) {
                  sycl::float2 _mean{0.0f, 0.0f};
                  for (int i = 0; (i + 1) < n_samples_template; i += 2) {
                    _mean += sycl::float2{local_data[local_index + i],
                                          local_data[local_index + i + 1]};
                  }
                  // If n_samples_template is odd we won't process the
                  // last element in the above loop, to include
                  // that we process the last element separately below
                  if (n_samples_template % 2 != 0) {
                    _mean[0] +=
                        local_data[local_index + n_samples_template - 1];
                  }
                  const float nstf = static_cast<float>(n_samples_template);
                  mean = (_mean[0] + _mean[1]) / nstf;
                }
                for (int i = 0; (i + 1) < n_samples_template; i += 2) {
                  const auto sample =
                      sycl::float2{local_data[local_index + i] - mean,
                                   local_data[local_index + i + 1] - mean};
                  const auto templ =
                      sycl::float2{local_templ[i], local_templ[i + 1]};
                  // Using MAD instead of FMA for speed, exchange if more
                  // accuracy is needed
                  numerator = sycl::fma(templ, sample, numerator);
                  sum_square = sycl::fma(sample, sample, sum_square);
                }
                // If n_samples_template is odd we won't process the
                // last element in the above loop, to include
                // that we process the last element separately below
                if (n_samples_template % 2 != 0) {
                  const float sample =
                      local_data[local_index + n_samples_template - 1] - mean;
                  const float temple = local_templ[n_samples_template - 1];
                  numerator[0] = sycl::fma(temple, sample, numerator[0]);
                  sum_square[0] = sycl::fma(sample, sample, sum_square[0]);
                }
                float denominator =
                    local_sum_sq[comp_offset] * (sum_square[0] + sum_square[1]);
                if (denominator > STABILITY_THRESHOLD) {
                  denominator = sycl::rsqrt(denominator);
                  // Using MAD instead of FMA for speed, exchange if more
                  // accuracy is needed
                  const float numer = numerator[0] + numerator[1];
                  cc_sum = sycl::mad((numer * denominator),
                                     local_weights[comp_offset], cc_sum);
                }
              }
            }
            if (idx.get_global_id()[0] < i_range.size()) {
              device_mem.cc_sum[index + cc_sum_offset] = cc_sum;
            }
          });
    });
  }
  // Since we only store CC_sum in device memory we need to copy it out
  for (int t = 0; t < n_templates; ++t) {
    const auto queue_index = t % queues.size();
    auto Q = queues[queue_index];
    auto device_mem = device_data[queue_index];
    const size_t cc_sum_offset = t * n_corr;
    const size_t network_offset = t * network_size;
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
    Q.copy<float>(&device_mem.cc_sum[cc_sum_offset + start_i],
                  &cc_sum[cc_sum_offset + start_i], stop_i - start_i);
  }
  // Free memory per queue
  for (size_t i = 0; i < queues.size(); i++) {
    auto Q = queues[i];
    // Wait for last submission of this queue
    Q.wait();
    auto dev_mem = device_data[i];
    sycl::free(dev_mem.templates, Q);
    sycl::free(dev_mem.sum_sq_templates, Q);
    sycl::free(dev_mem.data, Q);
    sycl::free(dev_mem.weights, Q);
    sycl::free(dev_mem.moveouts, Q);
    sycl::free(dev_mem.cc_sum, Q);
  }
  return 0;
} // matched_filter
