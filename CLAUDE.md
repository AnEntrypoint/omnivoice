# CLAUDE.md

## Web / ONNX export

`torch.nn.attention.flex_attention` is not ONNX-exportable. Pass `attn_implementation="eager"` to `from_pretrained` before calling `torch.onnx.export`.

WebGPU backend in `onnxruntime-web` does not accept `int64` tensors. The export wrappers in `export_onnx.py` accept `int32` inputs and cast to `int64` internally via `.long()` — do not change the wrapper input dtype.

The `attention_mask` in `OmniVoice.forward` is **full** (non-causal), not lower-triangular. Cond batch uses `[:seqLen, :seqLen] = True`, uncond uses `[:numTarget, :numTarget] = True` plus diagonal for padding positions. This is a masked-diffusion model, not autoregressive.

`OfflineAudioContext` in `audio.js` requires two passes (decode then resample) because the browser's `decodeAudioData` does not guarantee resampling to the target rate in a single call.

## Browser environment

`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers are required for `SharedArrayBuffer` (used by the ONNX WASM multi-threaded backend). Vite dev server sets these automatically; a production static server must set them manually.
