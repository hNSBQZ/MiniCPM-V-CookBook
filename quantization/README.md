# Model Quantization Guide

Reduce model storage and computational requirements for resource-constrained environments.

📖 [中文版本](./README_zh.md) | [Back to Main](../)

## Quantization Methods Comparison

| Method | Precision | Compression | Inference Speed | Memory Usage | Best For |
|--------|-----------|-------------|----------------|-------------|----------|
| **GGUF** | 4-bit/8-bit | High | Fast | Low | CPU inference, edge devices |
| **BNB** | 4-bit/8-bit | Medium | Medium | Medium | GPU training and inference |
| **AWQ** | 4-bit | High | Fast | Low | High-performance GPU inference |
| **GPTQ** | 4-bit/8-bit | High | Fast | Low | GPU inference, vLLM deployment |

## Method Details

### GGUF (GPT-Generated Unified Format)
- CPU-optimized quantization format
- Supports multiple precisions (Q4_0, Q4_1, Q8_0, etc.)
- Ultra-low memory usage, ideal for mobile devices and PCs
- Excellent cross-platform compatibility

### BNB (Bits and Bytes)
- Supports 4-bit and 8-bit quantization
- Seamless integration with transformers library
- Training-time quantization support, ideal for fine-tuning and research
- Minimal accuracy loss

### AWQ (Activation-aware Weight Quantization)
- Activation-aware weight quantization
- 4-bit precision with excellent performance retention
- Optimized for inference, suitable for production environments
- Ideal choice for high-throughput services

### GPTQ (Generative Pre-trained Transformer Quantization)
- Post-training weight quantization based on approximate second-order information
- Supports 4-bit and 8-bit precision
- Compatible with transformers and vLLM
- Excellent compression with minimal accuracy loss

## Selection Guide

- **Mobile/Edge Devices**: GGUF - CPU optimized, minimal memory footprint
- **Research & Development**: BNB - Easy integration, training-time quantization support
- **Production Environment**: AWQ - Excellent inference performance, suitable for large-scale deployment
- **Personal Use**: GGUF - CPU-friendly, low hardware requirements
- **vLLM Deployment**: GPTQ - Mature ecosystem, wide framework support
