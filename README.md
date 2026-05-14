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

Mentat is designed to be a completely independent, fast, and secure platform for running AI models locally. By building "from the ground up" in **Rust**, Mentat acts as the ultimate "reader" for AI models (weights), giving you absolute control over your intelligence infrastructure.

Our goal is to provide a system that:
- 🔒 **Runs 100% Locally:** No internet required, zero API costs, and total privacy for your data.
- 📦 **Is Easily Distributable:** Compile a single executable binary, bundle it with model weights, and run it anywhere without complex Python dependencies.
- ⚡ **Is Highly Optimized:** Leverages Rust's absolute memory control and native hardware acceleration to run models blazing fast on consumer hardware.
- 👑 **Empowers Sovereignty:** Load, run, and eventually fine-tune your own models, freeing yourself from locked-in corporate ecosystems.

## ✨ Key Features

- **Rust-Native Inference Engine:** Core tensor operations, Transformer blocks, and Mixture-of-Experts (MoE) logic built entirely in safe, concurrent Rust.
- **Agentic Capabilities:**
  - 🛠️ **Secure Execution:** Stateless, isolated environments for the model to execute code and solve problems.
  - 🌐 **Web Navigation:** Integrated tools for the model to search and extract information autonomously.
  - 📂 **Local Filesystem:** Atomic operations allowing the model to interact with local files safely.
- **Format Agnostic:** Designed to load standard weight formats natively (e.g., Safetensors, GGUF).
- **Advanced Tokenization:** High-performance BPE (Byte Pair Encoding) implementation to handle complex reasoning tokens and structured outputs.

## 🚀 Getting Started

### Prerequisites

You will need the Rust toolchain installed. If you haven't already, install it via `rustup`:

- [Install Rust](https://rustup.rs/)

### Installation

Clone the repository and build the engine using Cargo:

```bash
git clone https://github.com/mentat-ai/mentat
cd mentat
cargo build --release
```

### Usage (WIP)

Mentat is designed to be dead-simple to use. Once compiled, you will run your AI by pointing the engine to your model weights:

```bash
# Run a model in interactive mode
./target/release/mentat run --model ./models/my-model.safetensors --prompt "Explain quantum mechanics"

# Or serve it as a local API compatible with standard OpenAI clients
./target/release/mentat serve --port 8080
```

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mentat-ai/mentat&type=Date)](https://star-history.com/#mentat-ai/mentat&Date)
