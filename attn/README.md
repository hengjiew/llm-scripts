# flash-attention 2 vs flashinfer

The script compares flash-attention 2 and flashinfer.
flash-attention 2 from Dao-AILab requires that the paged KV cache block size must be divisible by 256
vLLM flash-attention 2 relaxes this requirement and allows the block size to be divisible by 16.

## Requirements
- [vLLM flash-attention 2](https://github.com/vllm-project/flash-attention)
```
pip install vllm_flash_attn
```
- [flashinfer](git@github.com:flashinfer-ai/flashinfer.git)
```
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4
```
## FP16/BF16
vllm_flash_attn-2.6.2 vs flashinfer-0.1.6+cu124torch2.4
[flashattn2.6.2_vs_flashinfer0.1.6](flashattn2.6.2_vs_flashinfer0.1.6)

## Notes
- Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
- Cannot use FlashAttention-2 backend for dtype other than torch.float16 or torch.bfloat16.
- Cannot use FlashAttention-2 backend for FP8 KV cache.
- Cannot use FlashAttention-2 backend for block size not divisible by 16.