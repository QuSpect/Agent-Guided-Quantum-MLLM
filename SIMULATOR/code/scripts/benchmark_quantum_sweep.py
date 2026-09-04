import json
import pathlib

from test_remote_quantum_gpu_v3 import benchmark


def main():
    results = []
    for batch in [4, 8, 16, 32, 64, 128]:
        results.append(benchmark(batch=batch, n_qubits=8, depth=2, repeats=50))
    payload = {
        "experiment": "EXP-001-batch-sweep",
        "results": results,
    }
    output = pathlib.Path(
        "./artifacts/EXP-001-batch-sweep.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

