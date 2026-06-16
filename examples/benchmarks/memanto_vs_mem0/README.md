# Memanto vs Mem0: The Great Agentic Memory Showdown

A rigorous benchmarking suite comparing [Memanto](https://github.com/moorcheh-ai/memanto) and [Mem0](https://github.com/mem0ai/mem0) agentic memory frameworks.

## Overview

This benchmark evaluates two fundamentally different approaches to agent memory:

| Aspect | Memanto | Mem0 |
|--------|---------|------|
| **Architecture** | Active companion agent with binary compression | LLM extraction + vector DB |
| **Write Path** | Direct storage (no LLM) | LLM extracts facts → vector indexing |
| **Read Path** | Exact semantic match | Approximate nearest neighbor (ANN) |
| **Storage** | ~128 bytes/memory (binary) | ~4,096 bytes/memory (Float32) |
| **Indexing Delay** | Zero | Seconds to minutes |
| **Token Overhead** | None on writes | ~2.5x input tokens per ingestion |

## Test Scenarios

### Scenario A: Context-Overhead & Latency Sprint
Ingests dense, shifting technical logs (Kubernetes, PostgreSQL, Redis, Kafka) and measures:
- **Ingestion latency** — time to store each memory
- **Token consumption** — total tokens consumed (extraction + storage)
- **Retrieval latency** — time to recall relevant memories
- **Recall accuracy** — whether correct memories are returned

### Scenario B: Shifting Persona & Temporal Tracking
Simulates an agent tracking evolving user preferences across 3 sessions:
- Phase 1: User prefers Python, dark mode, VS Code
- Phase 2: User switches to Go, Neovim
- Phase 3: User switches to light mode, abandons Rust goal

Measures whether frameworks correctly surface **current** preferences vs outdated ones.

## Metrics Collected

| Metric | How Measured | Why It Matters |
|--------|-------------|----------------|
| **Write Latency** | `time.perf_counter()` around write calls | Agent responsiveness |
| **Read Latency** | `time.perf_counter()` around read calls | Real-time agent interactions |
| **Input Tokens** | tiktoken `cl100k_base` encoding | API cost |
| **Output Tokens** | tiktoken counting on LLM responses | API cost |
| **Storage per Memory** | Framework-reported bytes | Infrastructure cost |
| **Recall Accuracy** | Keyword matching on known Q&A pairs | Memory reliability |
| **Answer Quality** | RAG answer vs ground truth keywords | End-to-end usefulness |

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/moorcheh-ai/memanto.git
cd memanto/examples/benchmarks/memanto_vs_mem0

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get API keys
#    Memanto: https://console.moorcheh.ai/api-keys (free, 100K ops)
#    Mem0: https://app.mem0.ai (free tier available)

# 4. Configure environment
export MEMANTO_API_KEY="your-memanto-key"
export MEM0_API_KEY=***"  # optional

# 5. Run the benchmark
python benchmark.py
```

## Output Files

| File | Description |
|------|-------------|
| `benchmark_report.md` | Human-readable report with tables and analysis |
| `benchmark_raw.json` | Raw metrics data for custom analysis |

## Environment Requirements

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.10 |
| Memanto | ≥ 1.0.0 |
| Mem0 | ≥ 0.1.0 |
| tiktoken | ≥ 0.7.0 |
| pandas | ≥ 2.0.0 |

## Host Environment

- **OS:** macOS (darwin) / Linux
- **Python:** 3.10+
- **Backend LLM:** Moorcheh (Memanto) / OpenAI (Mem0 platform)
- **Run timestamp:** Recorded in report header

## Reproducibility

To reproduce results:

1. Use the exact same API keys (or create new ones)
2. Install exact versions from `requirements.txt`
3. Run `python benchmark.py` — results are deterministic per API key
4. Compare your `benchmark_report.md` with ours

The benchmark uses `tiktoken` for consistent token counting across frameworks.

## Methodology Notes

- **Fair comparison:** Both frameworks use hosted backends (Moorcheh for Memanto, Mem0 Platform for Mem0)
- **Token counting:** Mem0's LLM extraction overhead is estimated at 2.5x input tokens (industry standard for fact extraction)
- **Storage estimates:** Memanto uses documented 128-byte binary compression; Mem0 uses standard 4096-byte Float32 vectors
- **Latency measurements:** Include network round-trip to hosted backends
- **Accuracy scoring:** Keyword-based matching against known ground truth answers

## Limitations

- API latency varies by network conditions and region
- Mem0 OSS mode (self-hosted) may perform differently from Platform mode
- Token estimates for Mem0 extraction are approximations
- Storage estimates are based on framework documentation, not direct measurement

## License

MIT
