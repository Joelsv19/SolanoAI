import torch


def main():
    print("=" * 45)
    print("        SolanoAI - Hardware Test")
    print("=" * 45)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        memory = torch.cuda.get_device_properties(0).total_memory
        memory_gb = memory / (1024 ** 3)

        print(f"VRAM: {memory_gb:.2f} GB")
        print("\nSolanoAI utilizara la GPU.")
    else:
        print("\nSolanoAI utilizara la CPU.")


if __name__ == "__main__":
    main()