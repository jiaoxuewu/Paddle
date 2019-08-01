/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <vector>
#include "paddle/fluid/framework/fleet/boxps.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

class BoxWrapper {
 public:
  virtual ~BoxWrapper() {}
  BoxWrapper() {}

  int PassBegin(const std::set<uint64_t>& feasgin_to_box) const;
  int PassEnd() const;
  int PullSparse(const Scope& scope, const paddle::platform::Place& place,
                 std::vector<std::string> para) {
    return 0;
  }
  int PushSparse(const Scope& scope, const paddle::platform::Place& place,
                 const std::vector<std::string> para) {
    return 0;
  }

  static std::shared_ptr<BoxWrapper> GetInstance() {
    if (nullptr == s_instance_) {
      // If main thread is guaranteed to init this, remove the lock
      static std::mutex mutex;
      std::lock_guard<std::mutex> lock(mutex);
      if (nullptr == s_instance_) {
        s_instance_.reset(new paddle::framework::BoxWrapper());
        // s_instance_->boxps_ptr_.reset(new paddle::boxps::FakeBoxPS());
        s_instance_->boxps_ptr_ = std::shared_ptr<paddle::boxps::BoxPS>(
            new paddle::boxps::FakeBoxPS());
        s_instance_->boxps_ptr_->init(
            9);  // Just hard code for embedding size now.
      }
    }
    return s_instance_;
  }

 private:
  static std::shared_ptr<paddle::boxps::BoxPS> boxps_ptr_;
  static std::shared_ptr<BoxWrapper> s_instance_;

 protected:
  static bool is_initialized_;  // no use now
};

}  // end namespace framework
}  // end namespace paddle
