"""Save the IBM Quantum account locally. RUN THIS YOURSELF in a terminal.

The API key is read with getpass (never echoed, never stored in shell history,
never shared with any agent or chat). It is written by qiskit-ibm-runtime to
~/.qiskit/qiskit-ibm.json, which stays on this machine.

Where to find the key: https://quantum.cloud.ibm.com -> log in -> dashboard ->
"API key" (create one if none exists). The Open plan instance is auto-detected;
if you were given an instance CRN, you can paste it at the second prompt,
otherwise just press Enter.

Usage (from D:\\woooorld):
    .venv-quantum\\Scripts\\python scripts\\save_ibm_account.py
"""
from getpass import getpass

from qiskit_ibm_runtime import QiskitRuntimeService


def main() -> None:
    token = getpass("Paste your IBM Quantum API key (input hidden): ").strip()
    if not token:
        raise SystemExit("Empty key - nothing saved.")
    instance = input("Instance CRN (optional, press Enter to auto-detect): ").strip() or None

    kwargs = {"channel": "ibm_quantum_platform", "token": token, "overwrite": True}
    if instance:
        kwargs["instance"] = instance
    QiskitRuntimeService.save_account(**kwargs)
    print("Saved. Verifying (may take ~10 s)...")

    service = QiskitRuntimeService()
    backends = service.backends()
    print(f"OK - account works. {len(backends)} backend(s) visible:")
    for backend in backends:
        print(f"  {backend.name}  ({backend.num_qubits} qubits)")


if __name__ == "__main__":
    main()
