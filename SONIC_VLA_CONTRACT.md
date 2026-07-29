# SONIC VLA Deployment Contract

This repository owns the real-robot sensor and controller boundary. All VLA
backends used here must implement the same contract; they do not emit robot
joint targets directly.

## Pipeline

```text
stereo camera + G1 state + tactile.body + prompt
  -> run_vla_inference.py
  -> GR00T PolicyServer :5550
  -> VLA backend
  -> 40 x 78 SONIC latent action chunk
  -> protocol-v4 ZMQ at 50 Hz
  -> gear_sonic_deploy C++ decoder/controller
  -> Unitree G1
```

## Policy Observation

The GR00T PolicyServer receives a batched observation:

```text
video.ego_view_left:  uint8 [B, 1, H, W, 3]
video.ego_view_right: uint8 [B, 1, H, W, 3]
state.*:              float32 [B, 1, D], canonical total width 46
tactile.tactile_raw:  uint8 [B, 1, 256]
language.*:           string [B, 1]
```

The 46 state dimensions follow the registered `unitree_g1_sonic` state-key
order. The tactile checkpoint used by `carry-bucket-stereo` was trained on one
legacy `body` packet. Therefore deployment subscribes to exactly
`tactile.body`; it must not concatenate the three newer
`vest/left_arm/right_arm` packets into an incompatible 768-dimensional value.

Stereo and tactile are required by default. A missing camera eye, missing
tactile frame, malformed 256-byte packet, or tactile data older than the
configured maximum age causes that inference request to be skipped. There is
no zero-fill fallback for a tactile checkpoint.

## Backend Wire Contract

Non-Isaac backends use the openpi-compatible websocket request:

```text
state:          float32 [46]
ego_view_left:  uint8 [H, W, 3]
ego_view_right: uint8 [H, W, 3]
prompt:         string
tactile:        uint8 [256]
```

The response is an already unnormalized, finite `float32 [40, 78]` array under
the `actions` key. Server handshake metadata declares contract version
`sonic_vla_v1`, dimensions, stereo keys, and whether tactile is required.

## Action Layout

```text
actions[:, 0:64]  = motion_token
actions[:, 64:71] = left_hand_joints
actions[:, 71:78] = right_hand_joints
```

`run_vla_inference.py` publishes one row at 50 Hz with protocol v4. Latency
compensation measures the complete policy request and skips expired rows from
the chunk before forwarding the remaining latent actions to the C++ SONIC
decoder/controller.
