"""
Method 5: AMP + torch.compile (Tensor Core utilization + kernel fusion)
torch.compile captures the FP16 compute graph post-autocast and fuses Conv->BN->ReLU
chains into single Tensor Core kernels, eliminating intermediate VRAM round-trips.
AMP halves the working-set size, which lets the fused kernels stay in L2 longer.
GradScaler handles FP16 gradient underflow. The combination is additive when compute
is the bottleneck: compile removes launch overhead and fusion stalls; AMP doubles
arithmetic throughput on Tensor Core hardware.
Compilation cost: ~60-120s first run (includes autocast graph variants), cached to disk.
"""
import os
import time
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import AI4MarsDataset
from unet import UNet
from utils import (
    set_seed, calculate_iou, CombinedLoss,
    DATA_ROOT, NUM_CLASSES, IGNORE_INDEX, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    CLASS_WEIGHTS, SEED, IMG_SIZE,
)

METHOD_NAME = "amp_compile"

os.makedirs(f"logs/{METHOD_NAME}", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


def run_validation(model, val_loader, criterion, device):
    model.eval()
    val_loss, val_miou_total = 0.0, 0.0
    class_ious_totals = [0.0] * NUM_CLASSES
    num_batches = len(val_loader)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            val_loss += loss.item()
            miou, class_ious = calculate_iou(outputs, targets, NUM_CLASSES, IGNORE_INDEX)
            val_miou_total += miou
            for i, iou in enumerate(class_ious):
                if not np.isnan(iou):
                    class_ious_totals[i] += iou

    n = num_batches if num_batches > 0 else 1
    return val_loss / n, val_miou_total / n, [t / n for t in class_ious_totals]


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{METHOD_NAME.upper()}] Running on {device}")

    print("Loading datasets...")
    train_dataset = AI4MarsDataset(root_dir=DATA_ROOT, split="train", augment=True)
    val_dataset = AI4MarsDataset(root_dir=DATA_ROOT, split="test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params / 1e6:.2f}M parameters")

    criterion = CombinedLoss(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, class_weights=CLASS_WEIGHTS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")

    # --- torch.compile wraps the model before AMP warmup so inductor sees the autocast graph ---
    print("Compiling model with torch.compile()...")
    compile_wall_start = time.perf_counter()
    model = torch.compile(model, backend="inductor", fullgraph=False)
    backend_used = "inductor"

    # Warmup under autocast so inductor compiles the FP16 kernel variants
    print("  Warming up under autocast (triggers FP16 kernel compilation)...")
    dummy_in = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=device)
    dummy_tgt = torch.zeros(BATCH_SIZE, IMG_SIZE, IMG_SIZE, dtype=torch.long, device=device)
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        dummy_out = model(dummy_in)
        dummy_loss = criterion(dummy_out, dummy_tgt)
    scaler.scale(dummy_loss).backward()
    torch.cuda.synchronize()
    compile_time_sec = time.perf_counter() - compile_wall_start
    optimizer.zero_grad()
    print(f"  Backend: {backend_used}")
    print(f"  Compilation + warmup took {compile_time_sec:.1f}s")

    compile_info = {"compilation_time_sec": compile_time_sec, "backend": backend_used}
    with open(f"logs/{METHOD_NAME}/compile_info.json", "w") as f:
        json.dump(compile_info, f, indent=4)

    # --- Resume from checkpoint if one exists ---
    start_epoch = 0
    best_miou = 0.0
    ckpt_path = f"checkpoints/{METHOD_NAME}_checkpoint.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_miou = ckpt["best_miou"]
        print(f"  Resumed: starting at epoch {start_epoch + 1}, best mIoU so far {best_miou:.4f}")

    # Fresh scheduler for the remaining epochs (handles both cold start and resume)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader),
        epochs=max(1, EPOCHS - start_epoch)
    )

    print(f"Starting {METHOD_NAME.upper()} training for {EPOCHS} epochs...")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        epoch_wall_start = time.perf_counter()
        train_loss = 0.0
        samples_processed = 0
        compute_events = []
        h2d_events = []

        for inputs_cpu, targets_cpu in train_loader:
            samples_processed += inputs_cpu.size(0)

            # --- Measure H2D transfer time ---
            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            h2d_start.record()
            inputs = inputs_cpu.to(device)
            targets = targets_cpu.to(device)
            h2d_end.record()
            h2d_events.append((h2d_start, h2d_end))

            # --- Measure fused FP16 compute (forward + backward + optimizer) ---
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            compute_start.record()

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            compute_end.record()
            compute_events.append((compute_start, compute_end))
            train_loss += loss.item()

        torch.cuda.synchronize()

        epoch_wall_time = time.perf_counter() - epoch_wall_start
        h2d_times_ms = [s.elapsed_time(e) for s, e in h2d_events]
        compute_times_ms = [s.elapsed_time(e) for s, e in compute_events]
        step_times_ms = [h + c for h, c in zip(h2d_times_ms, compute_times_ms)]
        throughput = samples_processed / epoch_wall_time
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0.0

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, avg_val_miou, avg_class_ious = run_validation(model, val_loader, criterion, device)

        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "mIoU": avg_val_miou,
            "per_class_iou": avg_class_ious,
            "epoch_wall_time_sec": epoch_wall_time,
            "epoch_gpu_time_ms": sum(compute_times_ms),
            "throughput_samples_sec": throughput,
            "peak_memory_mb": peak_memory_mb,
            "per_batch_gpu_times_ms": compute_times_ms,
            "per_batch_h2d_times_ms": h2d_times_ms,
            "per_batch_step_times_ms": step_times_ms,
            "avg_compute_time_ms": float(np.mean(compute_times_ms)),
            "avg_h2d_time_ms": float(np.mean(h2d_times_ms)),
            "avg_step_time_ms": float(np.mean(step_times_ms)),
            "compilation_time_sec": compile_time_sec,
            "backend": backend_used,
        }

        with open(f"logs/{METHOD_NAME}/epoch_{epoch + 1}.json", "w") as f:
            json.dump(epoch_stats, f, indent=4)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Wall: {epoch_wall_time:.2f}s | "
            f"Compute: {np.mean(compute_times_ms):.2f}ms/step | "
            f"H2D: {np.mean(h2d_times_ms):.2f}ms/step | "
            f"Throughput: {throughput:.1f} samp/s | "
            f"Mem: {peak_memory_mb:.0f}MB | "
            f"mIoU: {avg_val_miou:.4f}"
        )

        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            torch.save(model.state_dict(), f"checkpoints/{METHOD_NAME}_best.pth")

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_miou": best_miou,
        }, f"checkpoints/{METHOD_NAME}_checkpoint.pth")

    torch.save(model.state_dict(), f"checkpoints/{METHOD_NAME}_final.pth")
    print(f"\n[{METHOD_NAME.upper()}] Done. Best mIoU: {best_miou:.4f}")
    print(f"  Compile overhead: {compile_time_sec:.1f}s | Breakeven: {compile_time_sec / max(1, np.mean(step_times_ms) * 0.001):.0f} steps")


if __name__ == "__main__":
    main()
