from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import numpy as np


@dataclass
class EventSlice:
    events: torch.Tensor
    start_ts_us: Optional[float] = None
    end_ts_us: Optional[float] = None


class EventPillarsEncoder:
    """Encode events into 4-channel EventPillars tensor: NET, EET, AEP, NED.

    Input event format: [N, 4] with columns [x, y, t, p].
    - x, y: pixel indices
    - t: timestamp (any monotonic unit)
    - p: polarity, positive for ON and non-positive for OFF
    """

    def __init__(self, height: int, width: int):
        self.height = int(height)
        self.width = int(width)

    def encode(self, events: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        if events.ndim != 2 or events.shape[1] != 4:
            raise ValueError("events must be [N, 4] with columns [x, y, t, p]")

        if device is None:
            device = events.device

        events = events.to(device=device)
        if events.shape[0] == 0:
            return torch.zeros((4, self.height, self.width), dtype=torch.float32, device=device)

        x = events[:, 0].long().clamp(0, self.width - 1)
        y = events[:, 1].long().clamp(0, self.height - 1)
        t = events[:, 2].float()
        p = events[:, 3].float()

        flat_idx = (y * self.width + x).detach().cpu().numpy().astype("int64")
        hw = self.height * self.width

        t_np = t.detach().cpu().numpy().astype("float32")
        p_np = p.detach().cpu().numpy().astype("float32")

        t_min = float(t_np.min())
        t_range = max(float(t_np.max() - t_min), 1e-6)
        t_norm = (t_np - t_min) / t_range

        # 兼容旧版 PyTorch：使用 numpy 聚合替代 scatter_reduce_
        net_flat_np = np.zeros(hw, dtype=np.float32)
        np.maximum.at(net_flat_np, flat_idx, t_norm)

        eet_flat_np = np.ones(hw, dtype=np.float32)
        np.minimum.at(eet_flat_np, flat_idx, t_norm)

        pos_count_np = np.zeros(hw, dtype=np.float32)
        neg_count_np = np.zeros(hw, dtype=np.float32)
        np.add.at(pos_count_np, flat_idx, (p_np > 0).astype(np.float32))
        np.add.at(neg_count_np, flat_idx, (p_np <= 0).astype(np.float32))

        aep_flat_np = np.tanh(neg_count_np - pos_count_np).astype(np.float32)
        num_events_np = pos_count_np + neg_count_np
        max_count = max(float(num_events_np.max()), 1.0)
        ned_flat_np = (num_events_np / max_count).astype(np.float32)

        net = torch.from_numpy(net_flat_np).to(device=device).view(self.height, self.width)
        eet = torch.from_numpy(eet_flat_np).to(device=device).view(self.height, self.width)
        aep = torch.from_numpy(aep_flat_np).to(device=device).view(self.height, self.width)
        ned = torch.from_numpy(ned_flat_np).to(device=device).view(self.height, self.width)

        return torch.stack([net, eet, aep, ned], dim=0)

    def stack(self, slices: Iterable[torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
        encoded = [self.encode(e, device=device) for e in slices]
        if not encoded:
            raise ValueError("slices is empty")
        return torch.stack(encoded, dim=0)
