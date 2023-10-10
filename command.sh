CUDA_VISIBLE_DEVICES=0 python -u llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --wbits 8 --true-sequential --act-order --groupsize 128 --save llama7b-8bit-128g.pt > logs/8bits.log
CUDA_VISIBLE_DEVICES=0 python -u llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --wbits 2 --true-sequential --act-order --groupsize 128 --eval > logs/2bit-hardcode8bits-eval.log
CUDA_VISIBLE_DEVICES=0 python -u llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --wbits 3 --true-sequential --act-order --groupsize 128 --save-quantized-tensors --save llama7b-3bit-128g.pt --eval > logs/3bit-eval.log


CUDA_VISIBLE_DEVICES=0 python llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --wbits 4 --groupsize 128 --load llama7b-4bit-128g.pt --benchmark 2048 --check
CUDA_VISIBLE_DEVICES=0 python llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --wbits 8 --groupsize 128 --load llama7b-8bit-128g.pt --benchmark 2048 --check

CUDA_VISIBLE_DEVICES=0 python -u llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --eval > logs/16bit-eval.log

CUDA_VISIBLE_DEVICES=0 python llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --benchmark 2048 --check
CUDA_VISIBLE_DEVICES=0 python llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --benchmark 2048 --check --load-quantized-tensors llama7b-dyn-bits.pt

# write to dynbits.log
CUDA_VISIBLE_DEVICES=0 python -u llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --dynbits --wbits 4 --true-sequential --act-order --groupsize 128 --save-quantized-tensors --save llama7b-dyn-bits.pt > dynbits.log
CUDA_VISIBLE_DEVICES=0 python -u llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --dynbits --wbits 4 --true-sequential --act-order --groupsize 128 > dynbits.log

# console debug dynbits
CUDA_VISIBLE_DEVICES=0 python -u llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --outliers --dynbits --wbits 4 --true-sequential --act-order --groupsize 128 --save-quantized-tensors --save llama7b-dyn-bits-outliers-128g.pt > dynbits.log

# console debug dynbits save original tensor
CUDA_VISIBLE_DEVICES=0 python llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --dynbits --wbits 4 --true-sequential --act-order --groupsize 128 --save-quantized-tensors --save llama7b-dyn-bits.pt

# single tensor benchmark
CUDA_VISIBLE_DEVICES=0 python llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --wbits 4 --true-sequential --act-order --groupsize 128 --single-tensor-quant-benchmark --layer-idx 16 --tensor-name self_attn.q_proj

CUDA_VISIBLE_DEVICES=0 python llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --wbits 4 --groupsize 128 --single-tensor-quant-benchmark-woGPTQ --layer-idx 0 --tensor-name self_attn.q_proj