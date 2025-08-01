# vLLM Endpoint Service

[vLLM](https://github.com/vllm-project/vllm) is a fast and easy-to-use library for LLM inference and serving, it delivers state-of-the-art serving throughput with a set of advanced features such as PagedAttention, Continuous batching and etc.. Besides GPUs, vLLM already supported [Intel CPUs](https://www.intel.com/content/www/us/en/products/overview.html) and [Gaudi accelerators](https://habana.ai/products). This guide provides an example on how to launch vLLM serving endpoint on CPU and Gaudi accelerators.

## 🚀1. Set up Environment Variables

```bash
export LLM_ENDPOINT_PORT=8008
export host_ip=${host_ip}
export HF_TOKEN=${HF_TOKEN}
export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
export LLM_MODEL_ID="Intel/neural-chat-7b-v3-3"
```

For gated models such as `LLAMA-2`, you will have to pass the environment HF_TOKEN. Please follow this link [huggingface token](https://huggingface.co/docs/hub/security-tokens) to get the access token and export `HF_TOKEN` environment with the token.

## 🚀2. Set up vLLM Service

### 2.1 vLLM on CPU

First let's enable VLLM on CPU.

#### Build docker

```bash
bash ./build_docker_vllm.sh
```

The `build_docker_vllm` accepts one parameter `hw_mode` to specify the hardware mode of the service, with the default being `cpu`, and the optional selection can be `hpu`.

#### Launch vLLM service with scripts

```bash
bash ./launch_vllm_service.sh
```

If you want to customize the port or model_name, can run:

```bash
bash ./launch_vllm_service.sh ${port_number} ${model_name}
```

#### Launch vLLM service with docker compose

```bash
cd deployment/docker_compose
docker compose -f compose.yaml up vllm-server -d
```

### 2.2 vLLM on Gaudi

Then we show how to enable VLLM on Gaudi.

#### Build docker

```bash
bash ./build_docker_vllm.sh hpu
```

Set `hw_mode` to `hpu`.

#### Launch vLLM service on single node

1. Option 1: Use docker compose for quick deploy

```bash
cd deployment/docker_compose
docker compose -f compose.yaml up vllm-gaudi-server -d
```

2. Option 2: Use scripts to set parameters.

For small model, we can just use single node.

```bash
bash ./launch_vllm_service.sh ${port_number} ${model_name} hpu 1
```

Set `hw_mode` to `hpu` and `parallel_number` to 1.

The `launch_vllm_service.sh` script accepts 7 parameters:

- port_number: The port number assigned to the vLLM CPU endpoint, with the default being 8008.
- model_name: The model name utilized for LLM, with the default set to 'meta-llama/Meta-Llama-3-8B-Instruct'.
- hw_mode: The hardware mode utilized for LLM, with the default set to "cpu", and the optional selection can be "hpu".
- parallel_number: parallel nodes number for 'hpu' mode
- block_size: default set to 128 for better performance on HPU
- max_num_seqs: default set to 256 for better performance on HPU
- max_seq_len_to_capture: default set to 2048 for better performance on HPU

If you want to get more performance tuning tips, can refer to [Performance tuning](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#performance-tips).

#### Launch vLLM service on multiple nodes

For large model such as `meta-llama/Meta-Llama-3-70b`, we need to launch on multiple nodes.

```bash
bash ./launch_vllm_service.sh ${port_number} ${model_name} hpu ${parallel_number}
```

For example, if we run `meta-llama/Meta-Llama-3-70b` with 8 cards, we can use following command.

```bash
bash ./launch_vllm_service.sh 8008 meta-llama/Meta-Llama-3-70b hpu 8
```

### 2.3 vLLM with OpenVINO (on Intel CPU)

vLLM powered by OpenVINO supports all LLM models from [vLLM supported models list](https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md) and can perform optimal model serving on Intel GPU and all x86-64 CPUs with, at least, AVX2 support, as well as on both integrated and discrete Intel® GPUs (starting from Intel® UHD Graphics generation). OpenVINO vLLM backend supports the following advanced vLLM features:

- Prefix caching (`--enable-prefix-caching`)
- Chunked prefill (`--enable-chunked-prefill`)

#### Build Docker Image

To build the docker image for Intel CPU, run the command

```bash
bash ./build_docker_vllm_openvino.sh
```

Once it successfully builds, you will have the `vllm-openvino` image. It can be used to spawn a serving container with OpenAI API endpoint or you can work with it interactively via bash shell.

#### Launch vLLM service

For gated models, such as `LLAMA-2`, you will have to pass -e HUGGING_FACE_HUB_TOKEN=\<token\> to the docker run command above with a valid Hugging Face Hub read token.

Please follow this link [huggingface token](https://huggingface.co/docs/hub/security-tokens) to get an access token and export `HF_TOKEN` environment with the token.

```bash
export HF_TOKEN=<token>
```

To start the model server for Intel CPU:

```bash
bash launch_vllm_service_openvino.sh
```

#### Performance tips

vLLM OpenVINO backend environment variables

- `VLLM_OPENVINO_DEVICE` to specify which device utilize for the inference. If there are multiple GPUs in the system, additional indexes can be used to choose the proper one (e.g, `VLLM_OPENVINO_DEVICE=GPU.1`). If the value is not specified, CPU device is used by default.

- `VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON` enables U8 weights compression during model loading stage. By default, compression is turned off. You can also export model with different compression techniques using `optimum-cli` and pass exported folder as `<model_id>`

##### CPU performance tips

vLLM OpenVINO backend uses the following environment variables to control behavior:

- `VLLM_OPENVINO_KVCACHE_SPACE` to specify the KV Cache size (e.g, `VLLM_OPENVINO_KVCACHE_SPACE=40` means 40 GB space for KV cache), larger setting will allow vLLM running more requests in parallel. This parameter should be set based on the hardware configuration and memory management pattern of users.

- `VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8` to control KV cache precision. By default, FP16 / BF16 is used depending on platform.

- `VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON` to enable U8 weights compression during model loading stage. By default, compression is turned off.

To enable better TPOT / TTFT latency, you can use vLLM's chunked prefill feature (`--enable-chunked-prefill`). Based on the experiments, the recommended batch size is `256` (`--max-num-batched-tokens`)

OpenVINO best known configuration is:

    $ VLLM_OPENVINO_KVCACHE_SPACE=100 VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
        python3 vllm/benchmarks/benchmark_throughput.py --model meta-llama/Llama-2-7b-chat-hf --dataset vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --enable-chunked-prefill --max-num-batched-tokens 256

### 2.4 vLLM with ROCm (on AMD GPU)

#### Build docker image for ROCm vLLM

```bash
cd GenAIComps/comps/third_parties/vllm/src
docker build -f Dockerfile.amd_gpu -t opea/vllm-rocm:latest . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
```

#### Launch vLLM service with docker compose

```bash
cd GenAIComps/comps/third_parties/vllm/deployment/docker_compose
# IP port for vLLM service
export VLLM_SERVICE_PORT=8011
# HF token
export HF_TOKEN="your_hf_token"
# Cache dir
export HF_CACHE_DIR="./data"
# Model
export VLLM_LLM_MODEL_ID="Intel/neural-chat-7b-v3-3"
# Specify the number of GPUs used
export TENSOR_PARALLEL_SIZE=1
# Run deploy
docker compose -f compose.yaml up vllm-rocm-server -d
```

#### Checking ROCM vLLM service

```bash
curl http://${host_ip}:${VLLM_SERVICE_PORT}/v1/chat/completions \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"model": "Intel/neural-chat-7b-v3-3", "messages": [{"role": "user", "content": "What is Deep Learning?"}]}'
```

### 2.5 Query the service

And then you can make requests like below to check the service status:

```bash
curl http://${host_ip}:9009/v1/chat/completions \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "user", "content": "What is Deep Learning?"}]}'
```

## 🚀3. Set up LLM microservice

Then we warp the VLLM service into LLM microservice.

### Build docker

```bash
bash build_docker_microservice.sh
```

### Launch the microservice

```bash
bash launch_microservice.sh
```

### Consume the microservice

#### Check microservice status

```bash
curl http://${your_ip}:9000/v1/health_check\
  -X GET \
  -H 'Content-Type: application/json'

# Output
# {"Service Title":"opea_service@llm_vllm/MicroService","Service Description":"OPEA Microservice Infrastructure"}
```

#### Consume vLLM Service

User can set the following model parameters according to needs:

- max_tokens: Total output token
- stream(true/false): return text response in stream mode or non-stream mode

```bash
# stream mode
curl http://${your_ip}:9000/v1/chat/completions \
    -X POST \
    -d '{"model": "${model_name}", "messages": "What is Deep Learning?", "max_tokens":17}' \
    -H 'Content-Type: application/json'

curl http://${your_ip}:9000/v1/chat/completions \
    -X POST \
    -d '{"model": "${model_name}", "messages": [{"role": "user", "content": "What is Deep Learning?"}], "max_tokens":17}' \
    -H 'Content-Type: application/json'

#Non-stream mode
curl http://${your_ip}:9000/v1/chat/completions \
    -X POST \
    -d '{"model": "${model_name}", "messages": "What is Deep Learning?", "max_tokens":17, "stream":false}' \
    -H 'Content-Type: application/json'

```
