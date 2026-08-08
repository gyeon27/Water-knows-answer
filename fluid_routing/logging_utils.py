"""프레임별 상태 로깅. CSV로 프레임별 지표를, 종료 시 JSON으로 셀별 전환 요약을 남긴다."""

import csv
import json

import config as cfg


class Logger:
    def __init__(self, csv_path="fluid_routing_log.csv", summary_path="fluid_routing_summary.json"):
        self.csv_path = csv_path
        self.summary_path = summary_path
        self._file = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["frame", "n_stream", "n_splash", "n_pool", "pct_stream", "pct_splash", "pct_pool", "frame_time_ms"]
        )
        self._frame_times = []

    def log_frame(self, frame: int, state_counts: dict, frame_time_ms: float):
        n_stream = state_counts.get(cfg.STATE_STREAM, 0)
        n_splash = state_counts.get(cfg.STATE_SPLASH, 0)
        n_pool = state_counts.get(cfg.STATE_POOL, 0)
        total = max(1, n_stream + n_splash + n_pool)

        self._writer.writerow(
            [
                frame,
                n_stream,
                n_splash,
                n_pool,
                round(100.0 * n_stream / total, 3),
                round(100.0 * n_splash / total, 3),
                round(100.0 * n_pool / total, 3),
                round(frame_time_ms, 3),
            ]
        )
        self._frame_times.append(frame_time_ms)
        if frame % 30 == 0:
            self._file.flush()

    def write_summary(self, transition_count_np, num_frames: int):
        flicker_cells = [
            {"cell": int(i), "transition_count": int(c)}
            for i, c in enumerate(transition_count_np)
            if c >= cfg.FLICKER_COUNT_THRESHOLD
        ]
        flicker_cells.sort(key=lambda e: -e["transition_count"])

        summary = {
            "num_frames": num_frames,
            "avg_frame_time_ms": (sum(self._frame_times) / len(self._frame_times)) if self._frame_times else 0.0,
            "flicker_count_threshold": cfg.FLICKER_COUNT_THRESHOLD,
            "flicker_suspect_cells": flicker_cells[:50],
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def close(self):
        self._file.close()
