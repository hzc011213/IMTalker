import argparse

from tar_dataset import TarShardDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--min_frames", type=int, default=26)
    parser.add_argument("--landmark_pixel_scale", type=int, default=512)
    args = parser.parse_args()

    ds = TarShardDataset(
        shard_root=args.dataset_path,
        split=args.split,
        val_ratio=args.val_ratio,
        min_frames=args.min_frames,
        landmark_pixel_scale=args.landmark_pixel_scale,
    )

    print("dataset length:", len(ds))
    item = ds[0]
    for k, v in item.items():
        print(k, v.shape, v.dtype, float(v.min()), float(v.max()))


if __name__ == "__main__":
    main()
