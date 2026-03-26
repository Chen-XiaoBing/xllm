/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "qwen2_5_vision_layer.h"

namespace xllm {
namespace layer {

Qwen2_5_VisionLayerImpl::Qwen2_5_VisionLayerImpl(const ModelContext& context,
                                                 bool is_qwen3_style) {
  const auto& args = context.get_model_args();
  const auto& quant_config = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  int64_t dim = args.mm_hidden_size();
  int64_t mlp_intermediate_size = args.mm_intermediate_size();
  bool is_gated = true;
  attention_ = register_module("self_attn", Qwen2VisionAttention(context));
  norm1_ = register_module("norm1", RMSNorm(dim, args.rms_norm_eps(), options));
  norm2_ = register_module("norm2", RMSNorm(dim, args.rms_norm_eps(), options));

  if (is_qwen3_style) {
    norm1_->set_layernorm_mode();
    norm2_->set_layernorm_mode();
    is_gated = false;
  }
  LOG(INFO) << "is_qwen3_style: " << is_qwen3_style
            << ", is_gated: " << is_gated;

  bool has_bias = args.model_type() != "oxygenvlm";
  LOG(INFO) << "MLP with bias: " << has_bias
            << ", model_type:" << args.model_type();
  mlp_ = register_module("mlp",
                         DenseMLP(dim,
                                  args.mm_intermediate_size(),
                                  /*is_gated=*/is_gated,
                                  /*has_bias=*/has_bias,
                                  args.mm_hidden_act(),
                                  /*enable_result_reduction=*/true,
                                  quant_config,
                                  parallel_args.tp_group_,
                                  options));
  // LOG(INFO) << "===> dim: " << dim << " intermediate_size: " <<
  // args.mm_intermediate_size() << " mlp_: " << mlp_;
}

void Qwen2_5_VisionLayerImpl::load_state_dict(const StateDict& state_dict) {
  // LOG(INFO) << "===> start load_state_dict";
  attention_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
  mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
  norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
  // LOG(INFO) << "===> finish load_state_dict";
}

torch::Tensor Qwen2_5_VisionLayerImpl::forward(
    torch::Tensor& hidden_states,
    torch::Tensor& m_cos_pos,
    torch::Tensor& m_sin_pos,
    torch::Tensor& cu_seq_len,
    std::vector<int32_t>& cu_seq_len_vec,
    ModelInputParams& input_params,
    int node_id) {
  auto norm_output1 = std::get<0>(norm1_(hidden_states));
  // LOG(INFO) << "vision norm1: " << norm_output1;
  auto get_shape_str = [&](const at::Tensor& tensor) {
    std::stringstream ss;
    auto sizes = tensor.sizes();
    ss << "[";
    for (size_t i = 0; i < sizes.size(); ++i) {
      ss << sizes[i];
      if (i < sizes.size() - 1) ss << ", ";
    }
    ss << "]";
    return ss.str();
  };
  // LOG(INFO) << "===> norm_output1: " << get_shape_str(norm_output1);
  auto output = hidden_states + attention_(norm_output1,
                                           m_cos_pos,
                                           m_sin_pos,
                                           cu_seq_len,
                                           cu_seq_len_vec,
                                           input_params);
  // LOG(INFO) << "vision attn: " << output;
  auto norm_output2 = std::get<0>(norm2_(output));
  // LOG(INFO) << "vision norm2: " << norm_output2;
  output = output + mlp_(norm_output2);
  // LOG(INFO) << "vision output: " << output;
  // LOG(INFO) << "===> output: " << get_shape_str(output);
  return output;
}

}  // namespace layer
}  // namespace xllm
