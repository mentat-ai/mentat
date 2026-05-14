# Project Plan: Mentat

This document outlines the roadmap for building the Mentat sovereign inference engine and agentic platform in Rust.

## Phase 1: Foundation & Architecture
- [x] Initialize project structure (`src/`, `Cargo.toml`).
- [x] Define core structures for Data (Tensors, Matrices).
- [x] Implement basic tensor operations (Matrix multiplication, Activations) in Rust.
- [x] Setup CLI parsing and configuration management.

## Phase 2: Tokenization & Parsing
- [x] Implement a high-performance BPE (Byte Pair Encoding) tokenizer.
- [x] Support custom tokens for reasoning (e.g., think tags) and tool usage.
- [ ] Implement parsers for structured outputs and agentic interactions.

## Phase 3: The Inference Engine (The "Motor")
- [ ] Implement core Transformer block logic.
- [ ] Implement Mixture-of-Experts (MoE) routing logic.
- [ ] Build the Loader to read standard model weight formats (e.g., Safetensors, GGUF).
- [ ] Implement KV caching for efficient, fast decoding.

## Phase 4: Native Tools (The "Agentic" Layer)
- [ ] **Code Execution:** Implement a secure wrapper for isolated code execution.
- [ ] **Browser Tool:** Interface for web searching and content extraction.
- [ ] **Filesystem Operations:** Implement atomic file patching (Create, Update, Delete).

## Phase 5: Distribution & APIs
- [ ] Build the robust CLI tool for direct interaction (`cargo run -- run`).
- [ ] Implement a standard local HTTP API server for integrations (`cargo run -- serve`).
- [ ] Ensure the build process produces a static, easily distributable binary.

## Phase 6: Performance Optimization
- [ ] SIMD acceleration for Rust (using `std::simd` or specialized crates).
- [ ] Hardware acceleration support via FFI (e.g., Apple Metal, CUDA) for maximum speed.
- [ ] Deep benchmarking and memory profiling.

## Phase 7: Sovereignty & Fine-Tuning
- [ ] Data collection pipelines for local usage (opt-in, strictly private).
- [ ] Research and implement fine-tuning capabilities (e.g., LoRA) directly within the Rust ecosystem (e.g., via `candle` or `tch-rs`) to allow users to adapt models natively.
