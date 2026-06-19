import io
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from dataset import create_eye_mouth_mask


class TarShardDataset(Dataset):
    def __init__(
        self,
        shard_root,
        split="train",
        val_ratio=0.02,
        min_frames=26,
        landmark_pixel_scale=512,
    ):
        super().__init__()
        assert split in ["train", "val", "test"]

        self.shard_root = Path(shard_root)
        self.split = split
        self.val_ratio = val_ratio
        self.min_frames = min_frames
        self.pixel_scale = (landmark_pixel_scale, landmark_pixel_scale)

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        self._tar_cache = {}
        self.meta_list = self._build_index()

    def _norm_member(self, name: str) -> str:
        return name[2:] if name.startswith("./") else name

    def _frame_sort_key(self, frame_name: str):
        stem = Path(frame_name).stem
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return stem

    def _open_tar(self, tar_path: str):
        if tar_path not in self._tar_cache:
            self._tar_cache[tar_path] = tarfile.open(tar_path, "r")
        return self._tar_cache[tar_path]

    def _read_member_bytes(self, item):
        tf = self._open_tar(item["tar"])
        f = tf.extractfile(item["member"])
        if f is None:
            raise FileNotFoundError(f"Could not read tar member: {item['member']} from {item['tar']}")
        return f.read()

    def _read_image(self, frame_item):
        data = self._read_member_bytes(frame_item)
        return Image.open(io.BytesIO(data)).convert("RGB")

    def _read_landmarks(self, lmd_item):
        data = self._read_member_bytes(lmd_item)
        lines = data.decode("utf-8", errors="ignore").splitlines()

        frame_names = []
        landmarks = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = [p for p in line.split(" ") if p]
            if len(parts) < 2:
                continue

            frame_names.append(parts[0])

            coords = []
            for pair in parts[1:]:
                x, y = pair.split("_")
                coords.append((
                    float(x) / self.pixel_scale[0],
                    float(y) / self.pixel_scale[1],
                ))

            landmarks.append(coords)

        if len(landmarks) == 0:
            raise RuntimeError(f"No landmarks found in {lmd_item}")

        return frame_names, np.asarray(landmarks, dtype=np.float32)

    def _build_index(self):
        if not self.shard_root.exists():
            raise FileNotFoundError(f"shard_root does not exist: {self.shard_root}")

        chunk_dirs = sorted([p for p in self.shard_root.iterdir() if p.is_dir()])
        if not chunk_dirs:
            raise RuntimeError(f"No chunk directories found under {self.shard_root}")

        clip_to_frames = {}
        clip_to_lmd = {}

        for chunk_dir in tqdm(chunk_dirs, desc=f"Indexing tar shards [{self.split}]"):
            landmark_tar = chunk_dir / "landmarks.tar"
            if landmark_tar.exists():
                with tarfile.open(landmark_tar, "r") as tf:
                    for m in tf.getmembers():
                        if not m.isfile() or not m.name.endswith(".txt"):
                            continue

                        member = self._norm_member(m.name)
                        clip = Path(member).stem
                        clip_to_lmd[clip] = {
                            "tar": str(landmark_tar),
                            "member": m.name,
                        }

            frame_tars = sorted(chunk_dir.glob("frames_*.tar"))
            for frame_tar in frame_tars:
                with tarfile.open(frame_tar, "r") as tf:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        if not (m.name.endswith(".png") or m.name.endswith(".jpg")):
                            continue

                        member = self._norm_member(m.name)
                        parts = Path(member).parts
                        if len(parts) < 2:
                            continue

                        clip = parts[0]
                        frame_name = parts[-1]

                        clip_to_frames.setdefault(clip, []).append({
                            "tar": str(frame_tar),
                            "member": m.name,
                            "frame_name": frame_name,
                        })

        clips = sorted(set(clip_to_frames.keys()) & set(clip_to_lmd.keys()))
        if not clips:
            raise RuntimeError(f"No clips with both frames and landmarks found under {self.shard_root}")

        val_n = max(1, int(len(clips) * self.val_ratio))
        if self.split == "train":
            clips = clips[:-val_n]
        else:
            clips = clips[-val_n:]

        meta_list = []
        for clip in clips:
            frames = sorted(
                clip_to_frames[clip],
                key=lambda x: self._frame_sort_key(x["frame_name"]),
            )

            if len(frames) < self.min_frames:
                continue

            meta_list.append({
                "clip": clip,
                "frames": frames,
                "lmd": clip_to_lmd[clip],
            })

        if not meta_list:
            raise RuntimeError(
                f"No valid clips after filtering. "
                f"split={self.split}, min_frames={self.min_frames}, root={self.shard_root}"
            )

        print(f"[TarShardDataset] split={self.split}, clips={len(meta_list)}")
        return meta_list

    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, idx):
        meta = self.meta_list[idx]
        frames = meta["frames"]

        _, landmarks = self._read_landmarks(meta["lmd"])

        min_len = min(len(frames), len(landmarks))
        if min_len < 2:
            return self.__getitem__((idx + 1) % len(self.meta_list))

        frames = frames[:min_len]
        landmarks = landmarks[:min_len]

        f_id0, f_id1 = np.random.choice(min_len, size=2, replace=False)

        image_0 = self._read_image(frames[f_id0])
        image_1 = self._read_image(frames[f_id1])

        mask_eye_0, mask_mouth_0 = create_eye_mouth_mask(landmarks[f_id0], 512, 0, 2, 2)
        mask_eye_1, mask_mouth_1 = create_eye_mouth_mask(landmarks[f_id1], 512, 0, 2, 2)

        neg_idx = np.random.randint(len(self.meta_list))
        while neg_idx == idx:
            neg_idx = np.random.randint(len(self.meta_list))

        neg_meta = self.meta_list[neg_idx]
        neg_frames = neg_meta["frames"]
        _, neg_landmarks = self._read_landmarks(neg_meta["lmd"])

        neg_len = min(len(neg_frames), len(neg_landmarks))
        if neg_len < 1:
            neg_image = image_0
            neg_mask_eye = mask_eye_0
            neg_mask_mouth = mask_mouth_0
        else:
            neg_frame_id = np.random.randint(neg_len)
            neg_image = self._read_image(neg_frames[neg_frame_id])
            neg_mask_eye, neg_mask_mouth = create_eye_mouth_mask(
                neg_landmarks[neg_frame_id], 512, 0, 2, 2
            )

        return {
            "image_0": self.transform(image_0),
            "image_1": self.transform(image_1),
            "mask_eye_0": torch.tensor(mask_eye_0).permute(2, 0, 1),
            "mask_mouth_0": torch.tensor(mask_mouth_0).permute(2, 0, 1),
            "mask_eye_1": torch.tensor(mask_eye_1).permute(2, 0, 1),
            "mask_mouth_1": torch.tensor(mask_mouth_1).permute(2, 0, 1),
            "neg_image": self.transform(neg_image),
            "neg_mask_eye": torch.tensor(neg_mask_eye).permute(2, 0, 1),
            "neg_mask_mouth": torch.tensor(neg_mask_mouth).permute(2, 0, 1),
        }
