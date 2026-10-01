import sys


def main():
    lines = [f"python {sys.version.split()[0]}", f"exe {sys.executable}"]
    for module_name in ["numpy", "pandas", "sklearn", "matplotlib", "joblib"]:
        try:
            module = __import__(module_name)
            lines.append(f"{module_name} {getattr(module, '__version__', 'unknown')}")
        except Exception as error:  # noqa: BLE001 - diagnostic only
            lines.append(f"{module_name} MISSING ({error})")
    report = "\n".join(lines)
    with open("env_check_output.txt", "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
