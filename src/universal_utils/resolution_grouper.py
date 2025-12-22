import shutil
from pathlib import Path
from PIL import Image


class ImageResSorter:
    def __init__(self, image_dir):
        self.source_path = Path(image_dir)
        self.supported_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'
        }
        self._files = (
            f for f in self.source_path.iterdir()
            if f.is_file() and f.suffix.lower() in self.supported_extensions
        )

    def __iter__(self):
        return self

    def __next__(self):
        file_path = next(self._files)
        return self._process_image(file_path)

    def _get_dimensions(self, file_path):
        try:
            with Image.open(file_path) as img:
                return img.size
        except Exception:
            return None

    def _move_to_target(self, file_path, dimensions):
        width, height = dimensions
        target_dir = self.source_path / f"{width}x{height}"
        target_dir.mkdir(exist_ok=True)
        
        destination = target_dir / file_path.name
        if destination.exists():
            return False
            
        shutil.move(str(file_path), str(destination))
        return True

    def _process_image(self, file_path):
        dims = self._get_dimensions(file_path)
        if dims is None:
            return False
        return self._move_to_target(file_path, dims)

    def sort(self):
        if not self.source_path.exists():
            return

        results = [result for result in self]
        print(f"Sorted {sum(results)} images.")


if __name__ == "__main__":
    sorter = ImageResSorter("image_dir")
    sorter.sort()