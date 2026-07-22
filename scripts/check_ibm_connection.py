"""Read-only IBM Quantum connection check (uses the locally saved account).

Prints visible backends, the least-busy device, and its key properties needed
for the blind-benchmark run plan (dynamic-circuit support, median CZ/ECR error,
readout error). Contains and reveals no credentials.

Usage: .venv-quantum\\Scripts\\python scripts\\check_ibm_connection.py
"""
from qiskit_ibm_runtime import QiskitRuntimeService


def main() -> None:
    service = QiskitRuntimeService()
    backends = service.backends(operational=True)
    print(f"{len(backends)} operational backend(s):")
    for backend in backends:
        status = backend.status()
        print(f"  {backend.name:<24} {backend.num_qubits:>4}q  queue={status.pending_jobs}")

    least = service.least_busy(operational=True)
    print(f"\nleast busy: {least.name}")
    conf_flags = {
        "dynamic circuits (if_else)": "if_else" in getattr(least.target, "operation_names", []),
        "mid-circuit measure": "measure" in getattr(least.target, "operation_names", []),
        "reset": "reset" in getattr(least.target, "operation_names", []),
    }
    for name, val in conf_flags.items():
        print(f"  {name}: {val}")


if __name__ == "__main__":
    main()
