// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include <memory>

namespace paddle {
namespace framework {

std::shared_ptr<BoxWrapper> BoxWrapper::s_instance_ = nullptr;
std::shared_ptr<paddle::boxps::BoxPS> BoxWrapper::boxps_ptr_ = nullptr;

int BoxWrapper::PassBegin(const std::set<uint64_t>& feasgin_to_box) const {
  boxps_ptr_->PassBegin(feasgin_to_box);
  return 0;
}

int BoxWrapper::PassEnd() const {
  boxps_ptr_->PassEnd();
  return 0;
}

int BoxWrapper::PullSparse(const Scope& scope,
                           const paddle::platform::Place& place,
                           const std::vector<const uint64_t*>& keys,
                           const std::vector<float*>& values,
                           const std::vector<int64_t>& slot_lengths) {
  if (platform::is_cpu_place(place)) {
    VLOG(10) << "PaddleBox: PullSparse in CPUPlace";
    boxps_ptr_->PullSparseCPU(keys, values, slot_lengths);
  } else if (platform::is_gpu_place(place)) {
    VLOG(10) << "PaddleBox: PullSparse in CUDAPlace";
    platform::SetDeviceId(boost::get<platform::CUDAPlace>(place).GetDeviceId());
    auto slot_size = keys.size();
    std::vector<const uint64_t*> cpu_keys(slot_size);
    std::vector<float*> cpu_values(slot_size);
    for (auto i = 0; i < slot_size; ++i) {
      uint64_t* cpu_key_ptr = new uint64_t[slot_lengths[i]];
      cudaMemcpy(cpu_key_ptr, keys[i], slot_lengths[i] * sizeof(uint64_t),
                 cudaMemcpyDeviceToHost);
      cpu_keys[i] = cpu_key_ptr;

      cpu_values[i] = new float[slot_lengths[i] * 2];  // hidden_size_
    }
    boxps_ptr_->PullSparseCPU(cpu_keys, cpu_values, slot_lengths);
    for (auto i = 0; i < slot_size; ++i) {
      cudaMemcpy(values[i], cpu_values[i], slot_lengths[i] * 2 * sizeof(float),
                 cudaMemcpyHostToDevice);
      delete[] cpu_keys[i];
      delete[] cpu_values[i];
    }
  } else {
    VLOG(3)
        << "PaddleBox: PullSparse Only support CPUPlace and CUDAPlace now.\n";
    return 1;
  }
  return 0;
}

int BoxWrapper::PushSparseGrad(const Scope& scope,
                               const paddle::platform::Place& place,
                               const std::vector<const uint64_t*>& keys,
                               const std::vector<const float*>& grad_values,
                               const std::vector<int64_t>& slot_lengths) {
  if (platform::is_cpu_place(place)) {
    boxps_ptr_->PushSparseCPU(keys, grad_values, slot_lengths);
  } else if (platform::is_gpu_place(place)) {
    platform::SetDeviceId(boost::get<platform::CUDAPlace>(place).GetDeviceId());
    auto slot_size = keys.size();
    std::vector<const uint64_t*> cpu_keys(slot_size);
    std::vector<const float*> cpu_values(slot_size);
    for (auto i = 0; i < slot_size; ++i) {
      uint64_t* cpu_key_ptr = new uint64_t[slot_lengths[i]];
      cudaMemcpy(cpu_key_ptr, keys[i], slot_lengths[i] * sizeof(uint64_t),
                 cudaMemcpyDeviceToHost);
      cpu_keys[i] = cpu_key_ptr;

      float* cpu_value_ptr = new float[slot_lengths[i] * 2];  // hidden_size_
      cudaMemcpy(cpu_value_ptr, grad_values[i],
                 slot_lengths[i] * 2 * sizeof(float),
                 cudaMemcpyDeviceToHost);  // hidden_size_
      cpu_values[i] = cpu_value_ptr;
    }
    boxps_ptr_->PushSparseCPU(cpu_keys, cpu_values, slot_lengths);
    for (auto i = 0; i < slot_size; ++i) {
      delete[] cpu_keys[i];
      delete[] cpu_values[i];
    }
  } else {
    VLOG(3)
        << "PaddleBox: PullSparse Only support CPUPlace and CUDAPlace now.\n";
    return 1;
  }
  return 0;
}
}  // end namespace framework
}  // end namespace paddle
