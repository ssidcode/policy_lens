"""Usage: python -m eval.run_eval [--llm]   (default: offline deterministic mode)"""
import asyncio, json, sys
from pathlib import Path
from policylens.agent import answer_many

SET = json.loads((Path(__file__).parent / "eval_set.json").read_text())


def run(offline=True):
    recs = asyncio.run(answer_many([c["q"] for c in SET], offline=offline, log=False))
    cite_ok = hit_ok = tool_ok = n_cite = 0
    for c, r in zip(SET, recs):
        if c["expect"]:
            n_cite += 1
            hit_ok += c["expect"] in r["retrieved"]
            cite_ok += c["expect"] in r["citations"]
        tool_ok += all(t in r["tools"] for t in c["tools"])
        if c.get("should_refuse") and r["citations"] and not offline:
            tool_ok -= 1
    return {"n": len(SET), "retrieval_hit_rate": hit_ok / n_cite, "citation_accuracy": cite_ok / n_cite,
            "tool_selection_accuracy": tool_ok / len(SET),
            "avg_latency_s": round(sum(r["latency_s"] for r in recs) / len(recs), 2)}


if __name__ == "__main__":
    print(json.dumps(run(offline="--llm" not in sys.argv), indent=2))
