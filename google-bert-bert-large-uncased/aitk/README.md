# Google BERT optimization

This folder contains examples of Google BERT (bert-large-uncased) optimization using different workflows.

- QDQ for AMD NPU
- QDQ for Qualcomm NPU
- OpenVINO for Intel NPU
- Float downcasting for NVIDIA TRT for RTX GPU / DML for general GPU

## Google BERT optimization with QDQ for AMD NPU

This example performs Google BERT optimization with QDQ in one workflow. It performs the optimization pipeline:

- *HuggingFace Model -> Onnx Model -> Quantized Onnx Model*

**Configuration File**: `bert-large-uncased_qdq_amd.json`

**Key Features**:
- Optimized for deployment on an OpenVINO NPU backend (targets the OpenVINO execution provider).
- Uses the Hugging Face model "google-bert/bert-large-uncased" configured for feature extraction (fill‑mask).
- Calibrates quantization on the Wikipedia training split (up to 300 samples, batch size 1) using a custom transform script for transformer-specific quantization.
- Enforces static I/O with three inputs of length 128 and evaluates latency (avg and p90) with 20 warmup runs and 100 repeats; the original Hugging Face model is not evaluated.

## Google BERT optimization with QDQ for Qualcomm NPU

This example performs Google BERT optimization with QDQ in one workflow. It performs the optimization pipeline:

- *HuggingFace Model -> Onnx Model -> Quantized Onnx Model*

**Configuration File**: `bert-large-uncased_qdq_qnn.json`

**Key Features**:
- Hugging Face "google-bert/bert-large-uncased" for feature extraction, targeting a QNN-based NPU.
- Calibration/evaluation use XNLI (English) "premise" only, padded to 128, up to 10 samples, batch size 1.
- Convert to ONNX, fix shapes to [1,128], and apply graph/transformer optimizations.
- Static quantization (activations uint16, weights uint8); measure latency and throughput; original model not evaluated.

## Google BERT optimization with OpenVINO

This example performs Google BERT optimization with OpenVINO in one workflow for Intel NPU.

- *HuggingFace Model -> Onnx Model -> Quantized Onnx Model*

**Configuration File**: `bert-large-uncased_context_ov_static.json`

**Key Features**：
- HF "google-bert/bert-large-uncased" → converted for OpenVINO NPU with static three-input shape [1,128].
- Transformer static quantization calibrated on Wikipedia (300 samples); latency measured (avg & p90), original model not evaluated.

## Float downcasting for NVIDIA TRT for RTX GPU / DML for general GPU

It performs the optimization pipeline:

- *HuggingFace Model -> Onnx Model -> Float16 Onnx Model*

**Configuration File**: `bert-large-uncased_trtrtx.json`

## Dataset Information

### Quantization Datasets
- **QNN/AMD NPU**: Qualcomm QNN workflow uses XNLI (English) validation "premise" for calibration (≤10 samples, padded to 128); AMD QDQ workflow uses Wikipedia train split (300 samples) with custom preprocessing.
- **Intel NPU**: Uses Wikipedia train split (300 samples) with a custom transform script for transformer-specific quantization.

### Evaluation Datasets
- **Primary**: XNLI (English) validation split for NLI classification
- **Evaluation Metric**: Custom embedding accuracy for semantic similarity
- **Benchmark**: XNLI (Cross-lingual Natural Language Inference)


## Performance Evaluation Results
The following results are based on comprehensive evaluation using standard embedding benchmarks and performance metrics. All evaluations use the MTEB Banking77 dataset for consistency.

### Qualcomm NPU (QNN) Performance
 
| Metric | Value |
|--------|-------|
| **Latency (avg)** | 37.17 ms |
| **Latency (min)** | 47.16 ms |
| **Latency (max)** | 35.50 ms |
| **Throughput (avg)** | 27.67 tokens/sec |
| **Throughput (max)** | 28.34 tokens/sec |
| **Throughput (min)** | 25.13 tokens/sec |
 
### AMD NPU Performance
 
| Metric | Value |
|--------|-------|
| **Latency (avg)** | 51.28 ms |
| **Latency (min)** | 56.05 ms |
| **Latency (max)** | 45.37 ms |
| **Throughput (avg)** | 19.44 tokens/sec |
| **Throughput (max)** | 21.63 tokens/sec |
| **Throughput (min)** | 16.27 tokens/sec |
 
### Intel NPU Performance
 
| Metric | Value |
|--------|-------|
| **Latency (avg)** | 9.63 ms |
| **Latency (p90)** | 9.89 ms |
| **Similarity** | 0.9582 |

### TRT Performance
 
| Metric | Value |
|--------|-------|
| **Latency (avg)** | 3.11 ms |
| **Latency (min)** | 4.69 ms |
| **Latency (max)** | 2.55 ms |
| **Throughput (avg)** | 253.01 tokens/sec |
| **Throughput (max)** | 407.51 tokens/sec |
| **Throughput (min)** | 213.94 tokens/sec |

### DML Performance
 
| Metric | Value |
|--------|-------|
| **Latency (avg)** | 6.21 ms |
| **Latency (min)** | 6.75 ms |
| **Latency (max)** | 5.98 ms |
| **Throughput (avg)** | 151.48 tokens/sec |
| **Throughput (max)** | 166.66 tokens/sec |
| **Throughput (min)** | 109.40 tokens/sec |




## Notes

- The model used is `google-bert/bert-large-uncased` for feature extraction tasks
- Evaluation metrics focus on latency and throughput for feature extraction performance
- TBD values will be populated after running the optimization workflows and benchmarks