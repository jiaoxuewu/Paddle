/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/async_executor.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "gflags/gflags.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/executor_thread_worker.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/pybind/pybind.h"

namespace paddle {
namespace framework {

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,platform::dynload::ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void AsyncExecutor::PrepareReaders(std::vector<std::vector<std::shared_ptr<DataFeed>>>& readers,
                                   int ncards,
                                   int nreaders,
                                   const DataFeedDesc& data_feed_desc,
                                   const std::vector<std::string>& filelist) {
  readers.resize(ncards);
  for (size_t i = 0; i < readers.size(); ++i) {
    readers[i].resize(nreaders);
    for (int j = 0; j < nreaders; ++j) {
      readers[i][j] = DataFeedFactory::CreateDataFeed(data_feed_desc.name());
      readers[i][j]->Init(data_feed_desc);
    }
  }
  readers[0][0]->SetFileList(filelist);
}

void AsyncExecutor::InitRootScope(const ProgramDesc& program) {
  //auto& block = program.Block(0);

  //PADDLE_ENFORCE_NOT_NULL(
  //    root_scope_, "root_scope should be set before creating thread scope");

  //for (auto& var : block.AllVars()) {
  //  if (var->Persistable()) {
  //    auto* ptr = root_scope_->Var(var->Name());
  //    InitializeVariable(ptr, var->GetType());
  //  }
  //}
}

void AsyncExecutor::UpdateSyncFlag(int rank_id) {
  uint8_t* ptr = reinterpret_cast<uint8_t*>(&sync_flag_);
  ptr[rank_id] = 0;
  //fprintf(stderr, "r%d: %016lx\n", rank_id, sync_flag_);
  if (sync_flag_ == sync_signal_) {
    for (auto& worker : workers_) {
      worker->SetSyncSignal();
    }
    sync_flag_ = reset_sync_flag_;
    //fprintf(stderr, "r%d: %16lx (reset)\n", rank_id, sync_flag_);
  }
}

void AsyncExecutor::RunFromFile(const ProgramDesc& main_program,
                                const std::string& data_feed_desc_str,
                                const std::vector<std::string>& filelist,
                                const std::vector<std::string>& fetch_var_names,
                                const int ncards, const int nscopes, const int nreaders,
                                const int nemb_ff_threads, const int nemb_bp_threads,
                                const int nasync_steps) {
  auto& block = main_program.Block(0);
  for (auto var_name : fetch_var_names) {
    auto var_desc = block.FindVar(var_name);
    auto shapes = var_desc->GetShape();
    PADDLE_ENFORCE(shapes[shapes.size() - 1] == 1,
                   "var %s: Fetched var has wrong shape, "
                   "only variables with the last dimension size 1 supported",
                   var_name);
  }

  DataFeedDesc data_feed_desc;
  google::protobuf::TextFormat::ParseFromString(data_feed_desc_str, &data_feed_desc);

  int actual_ncards = ncards;
  int actual_nreaders = nreaders;
  int nfiles = filelist.size();
  PADDLE_ENFORCE(nfiles > 0, "File list cannot be empty");

  if (actual_ncards > nfiles) {
    VLOG(1) << "ncards = " << ncards << ", nfiles = " << nfiles << ". Changing ncards = " << nfiles;
    actual_ncards = nfiles;
    actual_nreaders = 1;
  }

  std::vector<std::vector<std::shared_ptr<DataFeed>>> readers;
  PrepareReaders(readers, actual_ncards, actual_nreaders, data_feed_desc, filelist);

  InitRootScope(main_program);

  std::shared_ptr<ncclUniqueId> nccl_id;
  ncclComm_t nccl_comms[actual_ncards];
  if (actual_ncards > 1) {
    nccl_id.reset(new ncclUniqueId);
    NCCLCHECK(platform::dynload::ncclGetUniqueId(nccl_id.get()));

    reset_sync_flag_ = 0;
    uint8_t* ptr = reinterpret_cast<uint8_t*>(&reset_sync_flag_);
    for (int i = 0; i < actual_ncards; ++i) {
      ptr[i] = ~0;
      fprintf(stderr, "%016lx\n", reset_sync_flag_);
    }
    sync_flag_ = reset_sync_flag_;

    NCCLCHECK(platform::dynload::ncclGroupStart());
    for (int i = 0; i < actual_ncards; ++i) {
      CUDACHECK(cudaSetDevice(i));
      NCCLCHECK(platform::dynload::ncclCommInitRank(nccl_comms + i, actual_ncards, *nccl_id, i));
    }
    NCCLCHECK(platform::dynload::ncclGroupEnd());
  }

  for (int i = 0; i < actual_ncards; ++i) {
    workers_.emplace_back(new ExecutorThreadWorker(actual_ncards, i, nscopes, nemb_ff_threads,
          nemb_bp_threads, nasync_steps, root_scope_, main_program, readers[i], fetch_var_names, nccl_comms + i, this));
  }

  // prepare thread resource here
  for (int thidx = 0; thidx < actual_ncards; ++thidx) {
    workers_[thidx]->CreateThreadResource();
  }

  std::vector<std::thread> threads;
  // start executing ops in multiple threads
  for (int thidx = 0; thidx < actual_ncards; ++thidx) {
    threads.push_back(
        std::thread(&ExecutorThreadWorker::TrainFiles, workers_[thidx].get()));
  }

  for (auto& t : threads) {
    t.join();
  }
  root_scope_->DropKids();

  if (actual_ncards > 1) {
    for (int i = 0; i < actual_ncards; ++i) {
      NCCLCHECK(platform::dynload::ncclCommDestroy(nccl_comms[i]));
    }
  }

  return;
}

}  // einit_modelnd namespace framework
}  // end namespace paddle
