# Qwen2.5-1.5B-Instruct optimization

This folder contains examples of Olive recipes for `Qwen2.5-1.5B-Instruct` optimization.

## INT4 AWQ Quantized Model Generation

The olive recipe `Qwen2.5-1.5B-Instruct_nvmo_int4_awq.json` produces INT4 AWQ quantized model using NVIDIA's TensorRT Model Optimizer toolkit.

### Setup

1. Install Olive with NVIDIA TensorRT Model Optimizer toolkit

    - Run following command to install Olive with TensorRT Model Optimizer.
    ```bash
    pip install olive-ai[nvmo]
    ```

2. Install suitable onnxruntime and onnxruntime-genai packages

    - Install the onnxruntime and onnxruntime-genai packages that have NvTensorRTRTXExecutionProvider support. Refer documentation for [NvTensorRtRtx execution-provider](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html/) to setup its dependencies/requirements. 
    - Note that by default, TensorRT Model Optimizer comes with onnxruntime-directml. And onnxrutime-genai-cuda package comes with onnxruntime-gpu. So, in order to use onnxruntime package with NvTensorRTRTXExecutionProvider support, one might need to uninstall existing other onnxruntime packages.
    - Make sure that at the end, there is only one onnxruntime package installed. Use command like following for validating the onnxruntime package installation.
    ```bash
    python -c "import onnxruntime as ort; print(ort.get_available_providers())"
    ```

3. Install additional requirements.

    - Install packages provided in requirements text file.
    ```bash
    pip install -r requirements-nvmo-awq.txt
    ```

### Recipe details

The olive recipe `Qwen2.5-1.5B-Instruct_nvmo_int4_awq.json` has 2 passes: (a) `ModelBuilder` and (b) `NVModelOptQuantization`. The `ModelBuilder` pass is used to generate the FP16 model for `NvTensorRTRTXExecutionProvider` (aka `NvTensorRtRtx` EP). Subsequently, the `NVModelOptQuantization` pass performs INT4 AWQ quantization to produce the 4-bit optimized model. In the quantization pass, execution-providers from the available/installed onnxruntime execution-providers is used for calibration. The field `calibration_providers` can be used to select any specific execution provider for calibration (assuming it is available/installed).
