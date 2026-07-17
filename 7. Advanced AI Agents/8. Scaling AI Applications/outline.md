```text
START — an LLM inference workload enters the system
  │
  ▼
[S1] Prepare the local inference environment
  ├── run on a Colab T4 GPU
  ├── install vLLM and benchmarking libraries
  ├── load Qwen2.5-0.5B-Instruct
  └── configure:
        model context length
        GPU memory utilization
        precision
        server port
  │
  ▼
[S2] Define what “performance” means
  ├── latency → time taken by one request
  ├── throughput → work completed per second
  ├── TTFT → time until the first generated output
  ├── ITL → delay between streamed output chunks
  ├── TPOT → average time per output token
  ├── E2E latency → total request completion time
  └── goodput → requests completed within an acceptable SLO
  │
  ▼
══════════════ OFFLINE INFERENCE ══════════════
  │
  ▼
[S3] Run vLLM directly inside Python
  ├── load the model into the vLLM engine
  ├── send prompts without an HTTP server
  └── establish a simple inference baseline
  │
  ▼
[S4] Experiment with batch size
  ├── send one or multiple prompts together
  ├── measure total execution time
  ├── calculate requests completed per second
  └── calculate generated tokens per second
  │
  ├── smaller batch
  │     lower waiting time
  │     weaker GPU utilization
  │
  └── larger batch
        better GPU utilization
        higher total throughput
        potentially more waiting per request
══════════════ ONLINE MODEL SERVING ══════════════
  │
  ▼
[S5] Convert the model into an inference service
  ├── remove the offline engine from GPU memory
  ├── start the vLLM HTTP server
  └── expose an OpenAI-compatible API endpoint
  │
  ▼
[S6] Call the local server as an application would
  ├── use the OpenAI Python SDK as the HTTP client
  ├── send chat-completion requests
  └── receive responses from the locally hosted model

No OpenAI-hosted model is used:

  OpenAI SDK
       │
       ▼
  local vLLM server
       │
       ▼
  Qwen model on the T4 GPU
  │
  ▼
[S7] Measure streaming behaviour

  request sent
       │
       ├──────────────► first chunk
       │                    TTFT
       │
       ├── chunk ── gap ── chunk
       │             ITL
       │
       └──────────────► final chunk
                          E2E latency

  └── divide completion time by output tokens
        to approximate TPOT
  │
  ▼
[S8] Repeat requests and inspect latency percentiles
  ├── P50 → typical user experience
  ├── P90/P95 → slower requests
  └── P99 → extreme tail latency

Average latency alone can hide a small number
of extremely slow requests.
  │
  ▼
[S9] Increase client concurrency
  ├── send several requests at the same time
  ├── measure requests and tokens per second
  ├── measure median and tail latency
  └── observe when the GPU becomes saturated
  │
  ├── low concurrency
  │     GPU may be underutilized
  │
  ├── moderate concurrency
  │     batching improves throughput
  │
  └── excessive concurrency
        requests queue up
        latency rises sharply
        throughput eventually stops improving
  │
  ▼

[S10] Benchmark using vLLM’s standard CLI
  ├── benchmark offline latency
  ├── benchmark the online server
  ├── control the incoming request rate
  ├── inspect TTFT, TPOT, ITL and E2E latency
  └── calculate goodput using explicit SLO limits
 
══════════════ MULTI-GPU SCALING ══════════════
  │
  ▼
[S11] Decide why the application needs more GPUs
  │
  ├── model fits on one GPU,
  │   but traffic is too high
  │          │
  │          ▼
  │     DATA PARALLELISM
  │     create complete model replicas
  │     and distribute requests between them
  │
  ├── model does not fit on one GPU,
  │   but fits inside one multi-GPU machine
  │          │
  │          ▼
  │     TENSOR PARALLELISM
  │     split matrix operations inside
  │     each transformer layer across GPUs
  │
  └── model must span several GPUs or machines
             │
             ▼
        PIPELINE PARALLELISM
        place different groups of layers
        on different GPUs or nodes
  │
  ▼
[S12] Understand tensor parallelism numerically

  Normal MLP:

      X → W1 → activation → W2 → output

  Tensor-parallel MLP:

      X
      ├── GPU 0 processes part of W1
      └── GPU 1 processes part of W1

      local activations
      ├── GPU 0 processes part of W2
      └── GPU 1 processes part of W2

      partial results
             │
             ▼
      communication and reduction
             │
             ▼
      same final output as the original MLP
  │
  ▼
[S13] Choose the deployment strategy

  Model fits on one GPU
      +
  traffic is small
      → no parallelism

  Model fits on one GPU
      +
  traffic is high
      → data-parallel replicas

  Model does not fit on one GPU
      +
  GPUs have a fast interconnect
      → tensor parallelism

  Model must span multiple nodes
      → tensor parallelism
        +
        pipeline parallelism

  Weak interconnect or uneven GPU memory
      → pipeline parallelism may be preferable
  │
  ▼
END RESULT

The notebook now demonstrates:

  • latency and throughput measurement
  • offline and online vLLM inference
  • batching and GPU utilization
  • prefill and decode behaviour
  • streaming metrics
  • percentile and tail latency
  • concurrency and server saturation
  • raw throughput versus goodput
  • request-rate benchmarking
  • data, tensor and pipeline parallelism
  • practical multi-GPU deployment decisions


CENTRAL IDEA

Scaling an AI application does not simply mean:

  “add more GPUs”

First measure:

  latency
      +
  throughput
      +
  tail behaviour
      +
  goodput
      +
  memory pressure

Then determine whether the real limitation is:

  request traffic
      +
  model size
      +
  GPU memory
      +
  GPU communication

Only then choose the appropriate serving
and parallelism strategy.
```
