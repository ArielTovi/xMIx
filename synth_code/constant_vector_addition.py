# xmix synthesized from ./vllm/activations_extractor/applications/xmix_examples/constant_vector_addition.xmix  (model: llama)
# assumes in scope: torch
#
# Excerpts are grouped by destination. Each '# >>> XMIX-APPLY file=… anchor=… <<<'
# tag is read by the apply tool to splice the block below it at that anchor;
# the '# >>> DESTINATION: … <<<' banner is the human-readable label.
# >>> XMIX-FOOTPRINT flag=w submodule=mlp.post layers=all <<<

# === SETUP (construct once — see destinations) ===

# --- user preamble (verbatim) ---

# >>> XMIX-APPLY file=vllm/v1/worker/gpu_model_runner.py anchor=runner_setup <<<
# >>> DESTINATION: gpu_model_runner — constructed once (e.g. in __init__/setup) <<<
self.arbitrary_vector = torch.full((self.hidden_size,), 1, device=self.device, dtype=self.model_config.dtype)
self.steer = SteeringVectorScaledAdder(self.arbitrary_vector, 8, 0)

# === INSTALL (after load_model) ===

# >>> XMIX-APPLY file=vllm/v1/worker/gpu_model_runner.py anchor=runner_postload <<<
# >>> DESTINATION: gpu_model_runner — after load_model (install hooks + wire buffers) <<<

# ./vllm/activations_extractor/applications/xmix_examples/constant_vector_addition.xmix:3  m.write(self.steer.run).layer("all").submodule("mlp.post")
# steer SteeringVectorScaledAdder on ALL layers at 'mlp.post' (flag 'w')
for layer in self.model.model.layers:
    layer.set_post_mlp_hook(self.steer.run, "w")
