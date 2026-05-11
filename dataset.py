import numpy as np
from pathlib import Path
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import IMG_SIZE

# Mask filenames are image filenames with an annotation suffix appended:
#   MER train:  STEM_merged6.png      (strip _merged6)
#   MER test:   STEM_16165_T0_merged.png  (strip _16165_T0_merged)
#   MSL:        STEM_15033_merged.png     (strip _15033_merged)
#                  or STEM_XXXX_15033_merged.png  (strip _15033_merged, keep _XXXX)
# Progressive rsplit('_') from the right until the image stem is found covers all cases.
#
# M2020 "M2020_GEO" labels use a completely different geological taxonomy (values 0-50)
# and are excluded — they are not compatible with the 5 AI4Mars terrain classes.


class AI4MarsDataset(Dataset):
    def __init__(self, root_dir, split="train", augment=False):
        """
        Args:
            root_dir: path to the ai4mars-dataset-merged-0.6 inner folder
            split:    'train' or 'test'
            augment:  apply random flips + colour jitter (training only)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.augment = augment and split == "train"
        self.samples: list[tuple[str, str]] = []

        img_ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
        if self.augment:
            img_ops.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2))
        img_ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        self.img_transform = transforms.Compose(img_ops)
        self.mask_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        ])

        self._build_dataset()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_image_index(self) -> dict[str, Path]:
        """Index every image file (not under labels/) by lowercase stem."""
        index: dict[str, Path] = {}
        for p in self.root_dir.rglob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg") and "labels" not in p.parts:
                index[p.stem.lower()] = p
        return index

    @staticmethod
    def _find_image(mask_stem: str, img_index: dict[str, Path]):
        """
        Strip trailing _word segments one at a time until the image stem is found.
        Handles all annotation suffix patterns without a fixed regex.
        Returns None if no image exists for this mask.
        """
        stem = mask_stem.lower()
        while True:
            img = img_index.get(stem)
            if img is not None:
                return img
            if '_' not in stem:
                return None
            stem = stem.rsplit('_', 1)[0]

    # ── dataset construction ──────────────────────────────────────────────────

    def _build_dataset(self):
        print(f"[AI4Mars] Building image index (split='{self.split}')...")
        img_index = self._build_image_index()
        print(f"  Images indexed: {len(img_index)}")

        # MER + MSL have standard train/test splits under labels/{split}/
        # M2020 GEO labels are excluded (wrong class taxonomy — values 0-50).
        mask_files = list(self.root_dir.rglob(f"labels/{self.split}/**/*.png"))

        matched = 0
        for mask_path in mask_files:
            img_path = self._find_image(mask_path.stem, img_index)
            if img_path is not None:
                self.samples.append((str(img_path), str(mask_path)))
                matched += 1

        skipped = len(mask_files) - matched
        print(f"  Masks found: {len(mask_files)} | Matched: {matched} | Skipped: {skipped}")
        if skipped:
            print(f"  ({skipped} masks skipped — images absent from this dataset download)")

        # M2020 NAV labels live in a flat folder with no train/test split.
        # Build matched pairs then do a deterministic 80/20 split by index.
        m2020_nav = self.root_dir / "m2020" / "labels" / "NAV"
        if m2020_nav.exists():
            nav_masks = sorted(m2020_nav.glob("*.png"))
            nav_pairs = []
            for mask_path in nav_masks:
                img_path = self._find_image(mask_path.stem, img_index)
                if img_path is not None:
                    nav_pairs.append((str(img_path), str(mask_path)))

            rng = random.Random(42)
            rng.shuffle(nav_pairs)
            cutoff = int(len(nav_pairs) * 0.8)
            if self.split == "train":
                split_pairs = nav_pairs[:cutoff]
            else:
                split_pairs = nav_pairs[cutoff:]

            self.samples.extend(split_pairs)
            print(f"  M2020 NAV: {len(nav_pairs)} matched pairs → {len(split_pairs)} for '{self.split}'")

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Paired spatial augmentation — same flip applied to both image and mask
        if self.augment:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask)).long()
        if mask.dim() == 3:
            mask = mask.squeeze(0)

        return image, mask


if __name__ == "__main__":
    dataset_path = "/home/abyhu/parallelproj/ai4mars-dataset-merged-0.6/ai4mars-dataset-merged-0.6"
    for split in ("train", "test"):
        ds = AI4MarsDataset(root_dir=dataset_path, split=split)
        if len(ds) > 0:
            img, msk = ds[0]
            print(f"[{split}] {len(ds)} pairs | image: {img.shape} | mask: {msk.shape} | classes: {msk.unique().tolist()}")
