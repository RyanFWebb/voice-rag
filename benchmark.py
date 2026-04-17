"""
benchmark.py - Compare RAG configurations on a fixed set of queries.

Runs each (config, query) pair and records stage-by-stage latency plus the
generated answer. Writes a CSV and prints a summary table.

Usage:
    python benchmark.py                          # default: all configs x all queries x 1 repeat
    python benchmark.py --repeats 3              # stable averages at 3x the runtime
    python benchmark.py --quick                  # 3 configs x 3 queries
    python benchmark.py --configs baseline,+greedy
    python benchmark.py --out results.csv
"""

import argparse
import csv
import time
from dataclasses import dataclass
from statistics import mean

import requests

import rag
import vector_store
from config import GEN_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT


TEST_QUERIES = [
    "What is the main conflict in Romeo and Juliet?",
    "How do Romeo and Juliet die?",
    "Who is Mercutio and what happens to him?",
    "What is the role of Friar Lawrence in the play?",
    "Why do the Montagues and Capulets hate each other?",
    "What does Juliet do to fake her death?",
]


@dataclass
class Config:
    name: str
    rerank: bool
    neighbor_window: int
    n_retrieve: int
    n_results: int
    temperature: float
    top_k: int | None

    def describe(self) -> str:
        return (
            f"rerank={self.rerank}, neighbors={self.neighbor_window}, "
            f"retrieve={self.n_retrieve}, results={self.n_results}, "
            f"temp={self.temperature}, top_k={self.top_k}"
        )


# Ordered: each config adds one change vs. the previous, then a few "what-if" sweeps.
CONFIGS = [
    Config("baseline",        rerank=False, neighbor_window=0, n_retrieve=5,  n_results=5, temperature=0.7, top_k=None),
    Config("+rerank",         rerank=True,  neighbor_window=0, n_retrieve=10, n_results=3, temperature=0.7, top_k=None),
    Config("+neighbors",      rerank=True,  neighbor_window=1, n_retrieve=10, n_results=3, temperature=0.7, top_k=None),
    Config("+greedy",         rerank=True,  neighbor_window=1, n_retrieve=10, n_results=3, temperature=0.0, top_k=20),
    Config("wider-neighbors", rerank=True,  neighbor_window=2, n_retrieve=10, n_results=3, temperature=0.0, top_k=20),
    Config("more-candidates", rerank=True,  neighbor_window=1, n_retrieve=20, n_results=3, temperature=0.0, top_k=20),
    Config("more-results",    rerank=True,  neighbor_window=1, n_retrieve=10, n_results=5, temperature=0.0, top_k=20),
]


def ollama_generate(prompt: str, temperature: float, top_k: int | None) -> str:
    """Direct Ollama call so the benchmark can vary temperature/top_k per config."""
    options = {"temperature": temperature, "num_ctx": 4096}
    if top_k is not None:
        options["top_k"] = top_k
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": GEN_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": options,
        },
        timeout=OLLAMA_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["response"]


def run_query(question: str, collection, cfg: Config) -> dict:
    timings = {}

    t0 = time.time()
    candidates = vector_store.query(question, collection, n_results=cfg.n_retrieve)
    timings["retrieve"] = time.time() - t0

    t0 = time.time()
    if cfg.rerank:
        reranked = rag.rerank(question, candidates, top_k=cfg.n_results)
    else:
        reranked = candidates[: cfg.n_results]
    timings["rerank"] = time.time() - t0

    t0 = time.time()
    contexts = vector_store.expand_neighbors(reranked, collection, window=cfg.neighbor_window)
    timings["expand"] = time.time() - t0

    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']} | "
        f"Pages: {','.join(str(p) for p in c['pages']) if c['pages'] else 'N/A'}]\n"
        f"{c['text']}"
        for c in contexts
    )
    prompt = rag.RAG_PROMPT.format(context=context_text, question=question)

    t0 = time.time()
    answer = ollama_generate(prompt, cfg.temperature, cfg.top_k)
    timings["llm"] = time.time() - t0

    timings["total"] = sum(timings.values())
    return {"answer": answer, "timings": timings, "n_contexts": len(contexts)}


def warm_up(collection):
    """Load embed model + LLM in Ollama so the first timed run isn't penalized."""
    print("Warming up models (embed + LLM + reranker)...")
    rag._get_reranker()
    vector_store.query("warm-up query", collection, n_results=1)
    ollama_generate("Hello.", temperature=0.0, top_k=None)
    print("Warm-up complete.\n")


def run_benchmark(configs, queries, repeats, out_path):
    collection = vector_store.get_collection()
    if collection.count() == 0:
        print("ERROR: vector store is empty. Run: python main.py --ingest")
        return

    total_runs = len(configs) * len(queries) * repeats
    print(f"Running {total_runs} queries "
          f"({len(configs)} configs x {len(queries)} queries x {repeats} repeat(s))\n")

    warm_up(collection)

    rows = []
    run_idx = 0
    for cfg in configs:
        print(f"=== {cfg.name} ===")
        print(f"    {cfg.describe()}\n")
        for q in queries:
            for rep in range(repeats):
                run_idx += 1
                print(f"  [{run_idx}/{total_runs}] {q[:62]}")
                result = run_query(q, collection, cfg)
                t = result["timings"]
                print(f"         retrieve={t['retrieve']:.2f}  rerank={t['rerank']:.2f}  "
                      f"expand={t['expand']:.2f}  llm={t['llm']:.2f}  total={t['total']:.2f}")
                rows.append({
                    "config":     cfg.name,
                    "query":      q,
                    "repeat":     rep,
                    "n_contexts": result["n_contexts"],
                    "retrieve_s": round(t["retrieve"], 3),
                    "rerank_s":   round(t["rerank"], 3),
                    "expand_s":   round(t["expand"], 3),
                    "llm_s":      round(t["llm"], 3),
                    "total_s":    round(t["total"], 3),
                    "answer":     result["answer"].replace("\n", " ").strip(),
                })
        print()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}\n")

    print_summary(rows, configs)


def print_summary(rows, configs):
    print("=== Latency summary (mean seconds over queries x repeats) ===\n")
    hdr = (f"{'config':<18}{'retrieve':>10}{'rerank':>10}"
           f"{'expand':>10}{'llm':>10}{'total':>10}{'vs base':>10}")
    print(hdr)
    print("-" * len(hdr))

    baseline_total = None
    for cfg in configs:
        cfg_rows = [r for r in rows if r["config"] == cfg.name]
        if not cfg_rows:
            continue
        avg = {k: mean(r[k] for r in cfg_rows) for k in
               ("retrieve_s", "rerank_s", "expand_s", "llm_s", "total_s")}
        if baseline_total is None:
            baseline_total = avg["total_s"]
            delta = "--"
        else:
            pct = (avg["total_s"] - baseline_total) / baseline_total * 100
            delta = f"{pct:+.0f}%"
        print(f"{cfg.name:<18}"
              f"{avg['retrieve_s']:>10.2f}"
              f"{avg['rerank_s']:>10.2f}"
              f"{avg['expand_s']:>10.2f}"
              f"{avg['llm_s']:>10.2f}"
              f"{avg['total_s']:>10.2f}"
              f"{delta:>10}")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark RAG configurations.")
    p.add_argument("--repeats", type=int, default=1, help="runs per (config, query) pair")
    p.add_argument("--quick", action="store_true",
                   help="shortcut: 3 configs (baseline, +rerank, +greedy) x 3 queries")
    p.add_argument("--configs", type=str, default=None,
                   help="comma-separated subset of config names to run")
    p.add_argument("--queries", type=int, default=None,
                   help="limit to the first N test queries")
    p.add_argument("--out", default="benchmark_results.csv")
    return p.parse_args()


def main():
    args = parse_args()

    configs = CONFIGS
    queries = TEST_QUERIES

    if args.quick:
        wanted = {"baseline", "+rerank", "+greedy"}
        configs = [c for c in CONFIGS if c.name in wanted]
        queries = TEST_QUERIES[:3]
    else:
        if args.configs:
            wanted = {name.strip() for name in args.configs.split(",")}
            configs = [c for c in CONFIGS if c.name in wanted]
            if not configs:
                print(f"No configs matched: {args.configs}")
                print(f"Available: {', '.join(c.name for c in CONFIGS)}")
                return
        if args.queries:
            queries = TEST_QUERIES[: args.queries]

    run_benchmark(configs, queries, args.repeats, args.out)


if __name__ == "__main__":
    main()
