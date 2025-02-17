from pathlib import Path
import matplotlib.pyplot as plt

BASE_IMAGE_PATH = Path() / "images"

def set_path(subfolder_name):
    global IMAGE_PATH
    IMAGE_PATH = BASE_IMAGE_PATH / subfolder_name
    IMAGE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Image path set to: {IMAGE_PATH}")


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGE_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    print(f"Figure saved at: {path}")

def main():
    print("Program started!")

if __name__ == "__main__":
    main()
