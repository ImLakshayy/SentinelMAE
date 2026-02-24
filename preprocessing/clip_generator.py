"""
preprocessing/clip_generator.py
"""

import argparse
import random
from pathlib import Path

import cv2
import torch
from tqdm import tqdm


def extract_clips(video_path, num_frames=16, clip_stride=8, image_size=224):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size))
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    cap.release()

    if len(frames) < num_frames:
        print(f"  [WARN] {video_path.name} too short ({len(frames)} frames), skipping.")
        return []

    clips = []
    start = 0
    while start + num_frames <= len(frames):
        clips.append(torch.stack(frames[start:start + num_frames], dim=0))
        start += clip_stride
    return clips


def build_dataset(raw_dir="data/raw", out_dir="data/processed",
                  num_frames=16, clip_stride=8, image_size=224,
                  val_split=0.2, seed=42, max_clips_per_video=10):
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    random.seed(seed)

    if not raw_path.exists():
        print(f"[ERROR] Folder not found: {raw_path}")
        print("  Create data/raw/ with class subfolders containing .mp4 files.")
        return

    class_dirs = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"[ERROR] No class folders found inside {raw_path}")
        return

    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    total = {"train": 0, "val": 0}

    for class_dir in class_dirs:
        videos = sorted(list(class_dir.glob("*.mp4")) +
                        list(class_dir.glob("*.avi")) +
                        list(class_dir.glob("*.mov")))
        if not videos:
            print(f"  [WARN] No videos in {class_dir.name}, skipping.")
            continue

        random.shuffle(videos)
        n_val = max(1, int(len(videos) * val_split))
        splits = {"val": videos[:n_val], "train": videos[n_val:]}
        print(f"\nClass '{class_dir.name}': {len(splits['train'])} train, {len(splits['val'])} val")

        for split, vid_list in splits.items():
            out_class = out_path / split / class_dir.name
            out_class.mkdir(parents=True, exist_ok=True)

            for vpath in tqdm(vid_list, desc=f"  [{split}] {class_dir.name}"):
                clips = extract_clips(vpath, num_frames, clip_stride, image_size)
                if len(clips) > max_clips_per_video:
                    clips = random.sample(clips, max_clips_per_video)
                for i, clip in enumerate(clips):
                    torch.save(clip, out_class / f"{vpath.stem}_clip{i:03d}.pt")
                total[split] += len(clips)

    print(f"\n Done!  Train clips: {total['train']}  |  Val clips: {total['val']}")
    print(f"Saved to: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--max_clips_per_video", type=int, default=10)
    args = parser.parse_args()
    build_dataset(args.raw_dir, args.out_dir, args.num_frames,
                  args.clip_stride, args.image_size, args.val_split,
                  max_clips_per_video=args.max_clips_per_video)