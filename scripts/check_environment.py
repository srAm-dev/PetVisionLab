import sys
import torch
import torchvision
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', help='file will be saved here')
    args = parser.parse_args()
    filedata = ''

    print(sys.executable)
    filedata += (sys.executable+'\n')

    print("PyTorch:", torch.__version__)
    filedata+=("PyTorch:"+torch.__version__+'\n')

    print("torchvision:", torchvision.__version__)
    filedata+=("torchvision:"+torchvision.__version__+'\n')

    print("PyTorch CUDA:", torch.version.cuda)
    filedata+=("PyTorch CUDA:"+str(torch.version.cuda)+'\n')

    print("CUDA available:", torch.cuda.is_available())
    filedata+=("CUDA available:"+str(torch.cuda.is_available())+'\n')

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        filedata+=("GPU:"+torch.cuda.get_device_name(0)+'\n')

        free, total = torch.cuda.mem_get_info(0)
        print(f"VRAM total/free: {total / 1024**3:.2f}/{free / 1024**3:.2f} GiB")
        filedata+=(f"VRAM total/free: {total / 1024**3:.2f}/{free / 1024**3:.2f} GiB"+'\n')

        x = torch.ones((2, 2), device="cuda")
        print("GPU matrix product:", x @ x)
        filedata+=("GPU matrix product:" + str(x @ x)+'\n')

    if args.output_dir:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        file = outdir / 'environment.txt'
        file.write_text(filedata, encoding="utf-8")

if __name__ == "__main__":
    main()