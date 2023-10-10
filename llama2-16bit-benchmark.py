tensor_names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj", "mlp.gate_proj"]

with open ("llama2-16bit-benchmark.sh", 'w') as f:
    for layer in range (32):
        for tensor_name in tensor_names:
            command = "CUDA_VISIBLE_DEVICES=0 python llama.py /home/v-wentaoni/workspace/llama-recipes/llama-2-7b-hf c4 --wbits 16 --true-sequential --act-order --groupsize 128 --single-tensor-quant-benchmark --layer-idx {} --tensor-name {} > /home/v-wentaoni/workspace/amlt/gptq_per_tensor_16bit/layer_{}.{}.txt".format (layer, tensor_name, layer, tensor_name)
            f.write (command + '\n')
