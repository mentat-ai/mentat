<div align="center">
<pre>
    __  ___           __        __ 
   /  |/  /__  ____  / /_____ _/ /_
  / /|_/ / _ \/ __ \/ __/ __ `/ __/
 / /  / /  __/ / / / /_/ /_/ / /_  
/_/  /_/\___/_/ /_/\__/\__,_/\__/  
</pre>
  <p><strong>A Sovereign, Rust-Native Inference Engine for High-Performance Reasoning Models.</strong></p>
  
  <p>
    <a href="https://github.com/mentat-ai/mentat/stargazers"><img src="https://img.shields.io/github/stars/mentat-ai/mentat?style=social" alt="Stars"></a>
    <a href="https://rustup.rs/"><img src="https://img.shields.io/badge/Language-Rust-orange.svg" alt="Built with Rust"></a>
    <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
    <a href="https://github.com/mentat-ai/mentat/issues"><img src="https://img.shields.io/github/issues/mentat-ai/mentat" alt="Issues"></a>
  </p>
</div>

---

## 🌟 Vision

Mentat is a completely independent, fast, and secure platform for running AI models locally. Built from the ground up in **Rust**, it extraction maximum performance from consumer hardware while ensuring absolute data privacy and architectural sovereignty.

## 🏗️ Technical Architecture

Mentat is designed with a modular, "purity-first" approach, separating the mathematical engine from the agentic capabilities.

### 1. The Core Engine (`src/tensor`)
The foundation of Mentat is a custom tensor library implemented in pure Rust. 
- **Tensor Ops:** Efficient implementation of `MatMul`, `Add`, and `Mul`.
- **Memory Management:** Leverages `memmap2` for zero-copy weight loading, allowing multi-gigabyte models to be loaded with minimal RAM overhead.
- **Precision Support:** Native support for `F32`, `F16`, and `BF16` (Brain-Float 16), ensuring compatibility with modern models like Llama 3 and GPT-OSS.

### 2. The Neural Brain (`src/model`)
A sovereign implementation of the Transformer architecture:
- **Transformer Blocks:** Modular blocks featuring `RMSNorm` (Root Mean Square Normalization) for stability.
- **Grouped-Query Attention (GQA):** Optimized attention mechanism for high-speed context processing.
- **Mixture of Experts (MoE):** Implementation of Gated Routing logic, enabling massive models to run efficiently by activating only a subset of parameters (Experts) per token.
- **KV Cache:** Advanced caching of Key-Value pairs to ensure O(1) inference time relative to sequence length.

### 3. The Communication Layer (`src/tokenizer`)
- **BPE Tokenizer:** A high-performance Byte Pair Encoding implementation for text-to-ID conversion.
- **Harmony Parser:** A specialized parser for structured outputs, capable of live-extracting reasoning chains (`<think>`) and agentic tool calls (`<python>`, `<browser>`) from the model's stream.

## 🗺️ Roadmap & Future Implementations

### Phase 4: Agentic Tools (Current Focus)
- [ ] **Secure Sandbox:** A WASM-based or Docker-isolated environment for executing model-generated Python code.
- [ ] **Sovereign Browser:** A headless navigation tool for real-time web research.
- [ ] **Atomic File Patcher:** Safe filesystem operations for direct codebase modifications.

### Phase 5: Distribution & APIs
- [ ] **OpenAI-Compatible API:** A local HTTP server that acts as a drop-in replacement for OpenAI endpoints.
- [ ] **Static Binaries:** Ensuring Mentat can be distributed as a single, dependency-free executable for Mac, Linux, and Windows.

### Phase 6: Hardware Acceleration
- [ ] **Apple Metal Support:** Native GPU acceleration for Apple Silicon via `cgo` or `metal-rs`.
- [ ] **CUDA/vLLM Integration:** High-performance kernels for NVIDIA hardware.

### Phase 7: Local Fine-Tuning
- [ ] **Native LoRA:** Implementation of Low-Rank Adaptation to allow users to "train" and adapt models to their own data locally without Python.

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/mentat-ai/mentat
cd mentat
cargo build --release
```

### Interactive Commands
Mentat provides a suite of tools for inspecting and testing models:

```bash
# 🔍 Inspect a model's internal architecture and tensors
cargo run --release -- inspect --model ./models/model.safetensors

# 📖 Test the BPE Tokenizer
cargo run --release -- tokenize "Hello, world!"

# 🧩 Test the Harmony Parser
cargo run --release -- parse "<think>Reasoning...</think> <python>print(1)</python>"
```

## 📜 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mentat-ai/mentat&type=Date)](https://star-history.com/#mentat-ai/mentat&Date)
