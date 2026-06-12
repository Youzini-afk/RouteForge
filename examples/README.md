# Examples

Examples will cover:

- recording HF Qwen3MoE routes;
- replaying HF Qwen3MoE R1/R2 routes;
- future DeepEP dispatch demos.

The current examples are lightweight sketches. They assume the caller has
already loaded a compatible Hugging Face Qwen3MoE model and prepared tokenizer
inputs. RouteForge records/replays the sparse block boundary where the model's
gate has produced top-k expert ids and route weights.

`output_router_logits=True` is not required and is not the replay source.
