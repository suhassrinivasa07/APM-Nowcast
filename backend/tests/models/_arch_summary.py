import torch
from models.apm_nowcast import APMNowcastModel

model = APMNowcastModel(T_in=4, T_out=4, latent_dim=24, hidden_dim=48, mc_dropout_rate=0.1, mc_samples=8)

# Total parameters
total = model.count_parameters(trainable_only=True)
print(f"Total trainable parameters: {total:,}")
print(f"Parameter budget check: {total} <= 3,000,000 -> {total <= 3_000_000}")
print()

# Per-component breakdown
components = {
    "Satellite Encoder": model.encoders["satellite"],
    "Radar Encoder": model.encoders["radar"],
    "Lightning Encoder": model.encoders["lightning"],
    "NWP Encoder": model.encoders["nwp"],
    "Adaptive Fusion": model.fusion,
    "Physics Conditioning": model.physics_conditioning,
    "Evolution Core": model.evolution_core,
    "Forecast Heads": model.forecast_heads,
}

header = "Component".ljust(25) + " | " + "Params".rjust(10)
print(header)
print("-" * 40)
for name, mod in components.items():
    p = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    print(name.ljust(25) + " | " + f"{p:>10,}")
print("-" * 40)
print("TOTAL".ljust(25) + " | " + f"{total:>10,}")
print()

# Forward pass test
B, T_in, T_out, H, W = 2, 4, 4, 64, 64
inputs = {
    "satellite": torch.randn(B, T_in, 3, H, W),
    "radar": torch.randn(B, T_in, 2, H, W),
    "lightning": torch.zeros(B, T_in, 2, H, W),
    "nwp": torch.randn(B, T_in, 4, H, W),
}
masks = torch.ones(B, T_in, 4)

outputs = model(inputs, masks, estimate_epistemic=True)
print("Output tensors:")
for key, val in outputs.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key}: {list(val.shape)}")
    elif isinstance(val, dict):
        print(f"  {key}: dict with keys {list(val.keys())}")

# VRAM estimation
vram32 = model.estimate_vram_mb(batch_size=4, patch_size=(64, 64), dtype=torch.float32)
vram16 = model.estimate_vram_mb(batch_size=4, patch_size=(64, 64), dtype=torch.float16)
print(f"\nEstimated VRAM (B=4, fp32): {vram32:.1f} MB")
print(f"Estimated VRAM (B=4, fp16): {vram16:.1f} MB")

# Sensor dropout test
print("\n--- Sensor Dropout Behavior ---")
masks_no_radar = masks.clone()
masks_no_radar[:, :, 1] = 0.0
inputs_no_radar = dict(inputs)
inputs_no_radar["radar"] = None

model.eval()
with torch.no_grad():
    out_drop = model(inputs_no_radar, masks_no_radar)
    radar_w = out_drop["modality_attention_weights"][:, 1]
    print(f"Radar attention weight (dropped): max={radar_w.max().item():.6f}, all_zero={bool((radar_w == 0.0).all())}")
    remaining = out_drop["modality_attention_weights"][:, [0, 2, 3]].sum(dim=1)
    print(f"Remaining modality weights sum: mean={remaining.mean().item():.6f}")
