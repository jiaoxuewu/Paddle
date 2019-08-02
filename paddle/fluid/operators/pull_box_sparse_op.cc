//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/pull_box_sparse_op.h"

namespace paddle {
namespace operators {

class PullBoxSparseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    // PADDLE_ENFORCE(ctx->HasInput("Ids"),
    // "Input(Ids) of PullBoxSparseOp should not be null.");
    // PADDLE_ENFORCE(ctx->HasOutput("Out"),
    // "Output(Out) of PullBoxSparseOp should not be null.");
    PADDLE_ENFORCE_GE(ctx->Inputs("Ids").size(), 1UL,
                      "Inputs(Ids) of PullBoxSparseOp should not be empty.");

    auto hidden_size = static_cast<int64_t>(ctx->Attrs().Get<int>("size"));
    auto all_ids_dim = ctx->GetInputsDim("Ids");
    const size_t n_ids = all_ids_dim.size();
    std::vector<framework::DDim> outs_dims;
    outs_dims.resize(n_ids);

    for (size_t i = 0; i < n_ids; ++i) {
      const auto ids_dims = all_ids_dim[i];
      int ids_rank = ids_dims.size();
      VLOG(5) << "the rank of the " << i << " ids is " << ids_rank << std::endl;
      PADDLE_ENFORCE_EQ(ids_dims[ids_rank - 1], 1,
                        "Shape error in %lu id, the last dimension of the "
                        "'Ids' tensor must be 1.",
                        i);

      // Make the output dim of all ids
      auto out_dim = framework::vectorize(
          framework::slice_ddim(ids_dims, 0, ids_rank - 1));
      out_dim.push_back(hidden_size);
      outs_dims[i] = framework::make_ddim(out_dim);
    }

    ctx->SetOutputsDim("Out", outs_dims);
    for (size_t i = 0; i < n_ids; ++i) {
      ctx->ShareLoD("Ids", "Out", i, i);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.MultiInput<framework::Tensor>("Ids")[0]->type(),
        ctx.device_context());
  }
};

class PullBoxSparseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids",
             "Input tensors with type int32 or int64 "
             "contains the ids to be looked up in BoxPS. "
             "The last dimension size must be 1.")
        .AsDuplicable();
    AddOutput("Out", "The lookup results tensors.").AsDuplicable();
    AddAttr<int>("size", "(int, the embedding hidden size").SetDefault(1);
    AddComment(R"DOC(
Pull Box Sparse Operator.

This operator is used to perform lookups on the BoxPS,
then concatenated into a dense tensor.

The input Ids can carry the LoD (Level of Details) information,
or not. And the output only shares the LoD information with input Ids.

)DOC");
  }
};

class PullBoxSparseGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("pull_box_sparse_grad");

    op->SetInput("Ids", Input("Ids"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

class PullBoxSparseOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.MultiInput<framework::Tensor>("Ids")[0]->type(),
        ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(pull_box_sparse, ops::PullBoxSparseOp,
                  ops::PullBoxSparseOpMaker, ops::PullBoxSparseGradOpDescMaker);

REGISTER_OPERATOR(pull_box_sparse_grad, ops::PullBoxSparseOpGrad);

REGISTER_OP_CPU_KERNEL(pull_box_sparse, ops::PullBoxSparseCPUKernel<float>,
                       ops::PullBoxSparseCPUKernel<int64_t>);
REGISTER_OP_CPU_KERNEL(pull_box_sparse_grad,
                       ops::PullBoxSparseGradKernel<float>,
                       ops::PullBoxSparseGradKernel<int64_t>);
