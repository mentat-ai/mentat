# Mentat

A sovereign, Rust-native inference engine for high-performance reasoning models. 

## Vision

Mentat is designed to be a completely independent, fast, and secure platform for running AI models locally. By building "from the ground up" in Rust, Mentat acts as the ultimate "reader" for AI models (weights). 

Our goal is to provide a system that:
- **Runs 100% Locally:** No internet required, no API costs, and total privacy for your data.
- **Is Easily Distributable:** Compile a single executable binary, bundle it with model weights, and run it anywhere without complex dependencies.
- **Is Highly Optimized:** Leverages Rust's absolute memory control and native hardware acceleration to run models efficiently on consumer hardware.
- **Empowers Sovereignty:** Allows users to load, run, and eventually fine-tune their own models, freeing them from locked-in ecosystems.

## Key Features

- **Rust-Native Inference Engine:** Core tensor operations, transformer blocks, and Mixture-of-Experts (MoE) logic built entirely in pure Rust.
- **Agentic Capabilities:**
  - **Secure Execution:** Stateless, isolated environments for the model to execute code and solve problems.
  - **Web Navigation:** Integrated tools for the model to search and extract information.
  - **Local Filesystem:** Atomic operations to allow the model to interact with local files safely.
- **Format Agnostic:** Designed to load standard weight formats natively.
- **Advanced Tokenization:** High-performance BPE (Byte Pair Encoding) implementation to handle complex reasoning tokens and structured outputs.

## Getting Started

### Prerequisites

- **Rust (Cargo)**: [Install Rust](https://rustup.rs/)

### Installation

```bash
git clone https://github.com/mentat-ai/mentat
cd mentat
cargo build --release
```

### Usage (Future)

Mentat is designed to be simple to use. Once compiled, you will run your AI by pointing the engine to your model weights:

```bash
./mentat run --model ./models/my-model-weights.bin --prompt "Hello, world!"
```

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
