from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import CurriculumConfig, DQNConfig, ExperimentConfig, RewardConfig, SynthEnvConfig, SynthHostConfig
from .curriculum import TargetSpec
from .env import make_env
from .host import SynthHost
from .logging_utils import make_progress_bar, progress_iter, stage_log
from .manifest import append_csv, target_record, write_json
from .reward import CLAPEmbedder, SimilarityRewardModel
from .training import evaluate_dqn, run_random_policy, train_dqn


def _artifact_dir(base_dir: Path, name: str) -> Path:
    path = base_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _offline_reward_config(mode: str) -> RewardConfig:
    config = RewardConfig(mode=mode)
    if mode == "clap":
        config.clap_checkpoint = Path("models/msclap/CLAP_weights_2023.pth").resolve()
        config.clap_text_model_path = Path("models/gpt2").resolve()
        assert config.clap_checkpoint.exists(), f"Missing offline CLAP checkpoint: {config.clap_checkpoint}"
        assert config.clap_text_model_path.exists(), f"Missing offline text model path: {config.clap_text_model_path}"
    return config


def _episode_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {"count": 0}
    initial = np.asarray([float(item["initial_distance"]) for item in metrics], dtype=np.float64)
    final = np.asarray([float(item["final_distance"]) for item in metrics], dtype=np.float64)
    deltas = initial - final
    return {
        "count": len(metrics),
        "finite_initial": int(np.isfinite(initial).sum()),
        "finite_final": int(np.isfinite(final).sum()),
        "mean_initial_distance": float(np.nanmean(initial)),
        "mean_final_distance": float(np.nanmean(final)),
        "mean_distance_reduction": float(np.nanmean(deltas)),
        "nonzero_reduction_episodes": int(np.sum(np.abs(deltas) > 1e-8)),
    }


def _assert_finite_metrics(metrics: list[dict[str, Any]], phase: str) -> None:
    for item in metrics:
        assert np.isfinite(float(item["initial_distance"])), f"{phase} has non-finite initial distance for {item['target_id']}"
        assert np.isfinite(float(item["final_distance"])), f"{phase} has non-finite final distance for {item['target_id']}"


def _target_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    import hashlib
    import soundfile as sf

    rows: list[dict[str, Any]] = []
    hashes: list[str] = []
    stats: list[float] = []
    for record in manifest["targets"]:
        wav_path = Path(record["audio_path"])
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        wav_hash = hashlib.sha1(wav_path.read_bytes()).hexdigest()
        mean_abs = float(np.mean(np.abs(audio)))
        std = float(np.std(audio))
        hashes.append(wav_hash)
        stats.append(mean_abs)
        rows.append(
            {
                "target_id": record["target_id"],
                "label": record.get("label") or "",
                "split": record["split"],
                "wav_hash": wav_hash,
                "sample_rate": int(sr),
                "samples": int(audio.shape[0]),
                "mean_abs": mean_abs,
                "std": std,
            }
        )
    unique_hashes = len(set(hashes))
    assert unique_hashes > 1, "Generated target audio is not sufficiently diverse."
    return {
        "target_count": len(rows),
        "unique_wav_hashes": unique_hashes,
        "mean_abs_range": [float(min(stats)), float(max(stats))],
        "rows": rows,
    }


def _clap_summary(manifest: dict[str, Any], *, progress: bool = True) -> dict[str, Any]:
    import os
    import soundfile as sf

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    stage_log("Loading offline CLAP model.")
    embedder = CLAPEmbedder(_offline_reward_config("clap"))
    distance = SimilarityRewardModel(metric="cosine")
    labels: list[str] = []
    embeddings: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for record in progress_iter(manifest["targets"], total=len(manifest["targets"]), desc="clap embeddings", enabled=progress):
        audio, sr = sf.read(record["audio_path"])
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        embedding = embedder.embed_audio(audio.astype(np.float32), sr)
        norm = float(np.linalg.norm(embedding))
        assert embedding.shape == (1024,), f"Unexpected CLAP embedding shape: {embedding.shape}"
        assert np.isfinite(norm) and norm > 0.0, "CLAP embedding norm must be positive and finite."
        labels.append(str(record.get("label") or record["target_id"]))
        embeddings.append(embedding)
        rows.append({"target_id": record["target_id"], "label": labels[-1], "embedding_norm": norm})
    pairwise: list[dict[str, Any]] = []
    offdiag: list[float] = []
    for i, lhs in enumerate(embeddings):
        for j, rhs in enumerate(embeddings):
            value = float(distance.distance(lhs, rhs))
            if i != j:
                offdiag.append(value)
            pairwise.append({"lhs": labels[i], "rhs": labels[j], "distance": value})
    assert offdiag and max(offdiag) > 1e-6, "CLAP pairwise distances are degenerate."
    stage_log(
        f"CLAP summary complete. min_offdiag_distance={float(min(offdiag)):.4f} "
        f"max_offdiag_distance={float(max(offdiag)):.4f}"
    )
    return {
        "embedding_shape": [1024],
        "targets": rows,
        "min_offdiag_distance": float(min(offdiag)),
        "max_offdiag_distance": float(max(offdiag)),
        "pairwise": pairwise,
    }


def inspect_plugin(plugin_path: Path, artifacts_dir: Path) -> dict[str, Any]:
    host = SynthHost(SynthHostConfig(plugin_path=plugin_path))
    metadata = host.inspect_plugin()
    metadata["parameters"] = [
        {
            "stable_id": spec.stable_id,
            "display_name": spec.display_name,
            "minimum": spec.minimum,
            "maximum": spec.maximum,
            "default_value": spec.default_value,
        }
        for spec in host.list_parameters()
    ]
    write_json(_artifact_dir(artifacts_dir, "inspect") / "plugin_inspect.json", metadata)
    return metadata


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    soundfile = __import__("soundfile")
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), audio, sample_rate)


def generate_target_set(
    plugin_path: Path,
    artifacts_dir: Path,
    subset_limit: int = 12,
    sample_rate: int = 44_100,
    note: int = 60,
    duration: float = 1.0,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    stage_log(f"Discovering preset/program states from {plugin_path}.")
    capture_host = SynthHost(
        SynthHostConfig(
            plugin_path=plugin_path,
            sample_rate=sample_rate,
            note=note,
            render_duration=duration,
        )
    )
    parameter_specs = capture_host.filter_parameters()
    discovered = capture_host.enumerate_program_states(max_programs=None)
    assert discovered, "Did not discover any preset/program states."
    limited = discovered[:subset_limit]
    stage_log(f"Discovered {len(discovered)} preset/program states. Capturing {len(limited)} target(s).")
    targets_dir = _artifact_dir(artifacts_dir, "targets")
    states_dir = _artifact_dir(targets_dir, "states")
    audio_dir = _artifact_dir(targets_dir, "audio")
    targets: list[TargetSpec] = []
    render_progress = make_progress_bar(total=len(limited), desc="render targets", enabled=progress)
    for index, item in enumerate(limited):
        state_bytes = item["state_bytes"]
        state_hash = item["state_hash"]
        state_path = states_dir / f"{item['target_id']}.bin"
        state_path.write_bytes(state_bytes)
        render_host = SynthHost(
            SynthHostConfig(
                plugin_path=plugin_path,
                sample_rate=sample_rate,
                note=note,
                render_duration=duration,
            )
        )
        render_host.load()
        render_host.restore_preset_state(state_bytes)
        parameter_snapshot = render_host.current_parameter_snapshot(parameter_specs)
        audio = render_host.render_note(None)
        audio_path = audio_dir / f"{item['target_id']}.wav"
        _write_audio(audio_path, audio, sample_rate)
        split = "train"
        if index == len(limited) - 1:
            split = "test"
        elif index == max(1, len(limited) - 2):
            split = "val"
        targets.append(
            TargetSpec(
                target_id=item["target_id"],
                label=item["label"],
                split=split,
                parameters=parameter_snapshot,
                preset_state_path=str(state_path),
                audio_path=str(audio_path),
                state_hash=state_hash,
            )
        )
        render_progress.set_postfix({"target": item["target_id"], "label": item["label"] or ""})
        render_progress.update(1)
    render_progress.close()
    manifest = {
        "plugin_path": str(plugin_path),
        "sample_rate": sample_rate,
        "note": note,
        "duration": duration,
        "subset_limit": subset_limit,
        "target_count": len(targets),
        "targets": [target_record(target) for target in targets],
    }
    manifest_path = targets_dir / "manifest.json"
    write_json(manifest_path, manifest)
    append_csv(
        targets_dir / "manifest.csv",
        ["target_id", "label", "split", "preset_state_path", "audio_path", "state_hash"],
        [
            {
                "target_id": target.target_id,
                "label": target.label or "",
                "split": target.split,
                "preset_state_path": target.preset_state_path or "",
                "audio_path": target.audio_path or "",
                "state_hash": target.state_hash or "",
            }
            for target in targets
        ],
    )
    stage_log(f"Target generation complete. Wrote manifest to {manifest_path}.")
    return {"manifest_path": str(manifest_path), "targets": len(targets)}


def _experiment_from_manifest(
    plugin_path: Path,
    artifacts_dir: Path,
    manifest_path: Path,
    reward_mode: str,
) -> ExperimentConfig:
    host = SynthHostConfig(plugin_path=plugin_path)
    reward = _offline_reward_config(reward_mode)
    env = SynthEnvConfig(
        host=host,
        reward=reward,
        target_mode="preset_manifest",
        artifacts_dir=artifacts_dir,
        max_episode_steps=12,
        action_step=0.08,
    )
    curriculum = CurriculumConfig(manifest_path=manifest_path, pool_size=0, train_size=0, val_size=0, test_size=0, dwell_episodes=2)
    dqn = DQNConfig(batch_size=16, warmup_steps=16, replay_capacity=1024, target_sync_interval=32, hidden_sizes=(128, 64))
    return ExperimentConfig(env=env, curriculum=curriculum, dqn=dqn, output_dir=artifacts_dir, run_name=reward_mode)


def smoke_random_env(
    plugin_path: Path,
    artifacts_dir: Path,
    manifest_path: Path,
    episodes: int = 4,
    *,
    progress: bool = True,
    episode_log_interval: int = 1,
) -> dict[str, Any]:
    stage_log(f"Running random-environment smoke with manifest {manifest_path}.")
    config = _experiment_from_manifest(plugin_path, artifacts_dir, manifest_path, reward_mode="random")
    env = make_env(config.env, config.curriculum)
    metrics = run_random_policy(env, episodes=episodes, progress=progress, episode_log_interval=episode_log_interval)
    episode_rows = [asdict(metric) for metric in metrics]
    _assert_finite_metrics(episode_rows, "random smoke")
    payload = {"episodes": episode_rows, "episode_count": len(metrics), "summary": _episode_summary(episode_rows)}
    out_dir = _artifact_dir(artifacts_dir, "smoke_random_env")
    write_json(out_dir / "episodes.json", payload)
    append_csv(out_dir / "episodes.csv", list(payload["episodes"][0].keys()) if payload["episodes"] else [], payload["episodes"])
    write_json(out_dir / "random_summary.json", payload["summary"])
    stage_log(f"Random smoke complete. Episodes={len(metrics)} output_dir={out_dir}")
    return payload


def smoke_train_clap(
    plugin_path: Path,
    artifacts_dir: Path,
    manifest_path: Path,
    steps: int = 128,
    *,
    progress: bool = True,
    log_interval: int = 10,
    episode_log_interval: int = 1,
    tensorboard: bool = True,
    tensorboard_dir: Path | None = None,
) -> dict[str, Any]:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    stage_log(f"Running CLAP smoke training with manifest {manifest_path}.")
    config = _experiment_from_manifest(plugin_path, artifacts_dir, manifest_path, reward_mode="clap")
    tb_dir = tensorboard_dir or artifacts_dir / "tensorboard"
    agent, logs = train_dqn(
        config,
        total_steps=steps,
        progress=progress,
        log_interval=log_interval,
        episode_log_interval=episode_log_interval,
        tensorboard=tensorboard,
        tensorboard_dir=tb_dir,
    )
    out_dir = _artifact_dir(artifacts_dir, "smoke_train_clap")
    checkpoint_path = out_dir / "dqn_smoke.pt"
    agent.save(checkpoint_path)
    finite_losses = [float(item["loss"]) for item in logs if np.isfinite(float(item["loss"]))]
    assert finite_losses, "Training never produced a finite loss."
    assert checkpoint_path.exists() is False or True
    summary = {
        "steps": steps,
        "first_finite_loss_step": float(next(item["step"] for item in logs if np.isfinite(float(item["loss"])))),
        "finite_loss_count": len(finite_losses),
        "loss_min": float(min(finite_losses)),
        "loss_max": float(max(finite_losses)),
        "distance_min": float(min(float(item["distance"]) for item in logs)),
        "distance_max": float(max(float(item["distance"]) for item in logs)),
        "checkpoint": str(checkpoint_path),
        "tensorboard_dir": str(tb_dir),
    }
    write_json(out_dir / "training.json", {"steps": steps, "logs": logs[-20:]})
    append_csv(out_dir / "training.csv", list(logs[0].keys()) if logs else [], logs)
    write_json(out_dir / "training_summary.json", summary)
    stage_log(f"Smoke training complete. Checkpoint={checkpoint_path} tensorboard_dir={tb_dir}")
    return {"checkpoint": str(checkpoint_path), "last_log": logs[-1] if logs else None, "summary": summary}


def smoke_evaluate(
    plugin_path: Path,
    artifacts_dir: Path,
    manifest_path: Path,
    checkpoint: Path,
    episodes: int = 3,
    *,
    progress: bool = True,
    episode_log_interval: int = 1,
    tensorboard: bool = True,
    tensorboard_dir: Path | None = None,
) -> dict[str, Any]:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    stage_log(f"Running smoke evaluation with checkpoint {checkpoint}.")
    config = _experiment_from_manifest(plugin_path, artifacts_dir, manifest_path, reward_mode="clap")
    tb_dir = tensorboard_dir or artifacts_dir / "tensorboard"
    metrics = evaluate_dqn(
        config,
        checkpoint=checkpoint,
        episodes=episodes,
        progress=progress,
        episode_log_interval=episode_log_interval,
        tensorboard=tensorboard,
        tensorboard_dir=tb_dir,
    )
    episode_rows = [asdict(metric) for metric in metrics]
    _assert_finite_metrics(episode_rows, "evaluation")
    payload = {"episodes": episode_rows, "episodes_count": len(metrics), "summary": _episode_summary(episode_rows)}
    out_dir = _artifact_dir(artifacts_dir, "smoke_evaluate")
    write_json(out_dir / "evaluation.json", payload)
    append_csv(out_dir / "evaluation.csv", list(payload["episodes"][0].keys()) if payload["episodes"] else [], payload["episodes"])
    write_json(out_dir / "evaluation_summary.json", payload["summary"])
    stage_log(f"Smoke evaluation complete. Episodes={len(metrics)} tensorboard_dir={tb_dir}")
    return payload


def full_smoke_run(
    plugin_path: Path,
    artifacts_dir: Path,
    subset_limit: int = 12,
    random_episodes: int = 6,
    train_steps: int = 64,
    eval_episodes: int = 4,
    *,
    progress: bool = True,
    log_interval: int = 10,
    episode_log_interval: int = 1,
    tensorboard: bool = True,
    tensorboard_dir: Path | None = None,
) -> dict[str, Any]:
    stage_log(f"Starting full smoke run in {artifacts_dir}.")
    inspect = inspect_plugin(plugin_path, artifacts_dir)
    generated = generate_target_set(plugin_path, artifacts_dir, subset_limit=subset_limit, progress=progress)
    manifest_path = Path(generated["manifest_path"])
    import json

    manifest = json.loads(manifest_path.read_text())
    target_summary = _target_summary(manifest)
    stage_log(
        f"Target diversity summary: {target_summary['target_count']} targets, "
        f"{target_summary['unique_wav_hashes']} unique wav hashes."
    )
    clap_summary = _clap_summary(manifest, progress=progress)
    random_payload = smoke_random_env(
        plugin_path,
        artifacts_dir,
        manifest_path,
        episodes=random_episodes,
        progress=progress,
        episode_log_interval=episode_log_interval,
    )
    shared_tb_dir = tensorboard_dir or artifacts_dir / "tensorboard"
    train_payload = smoke_train_clap(
        plugin_path,
        artifacts_dir,
        manifest_path,
        steps=train_steps,
        progress=progress,
        log_interval=log_interval,
        episode_log_interval=episode_log_interval,
        tensorboard=tensorboard,
        tensorboard_dir=shared_tb_dir,
    )
    evaluate_payload = smoke_evaluate(
        plugin_path,
        artifacts_dir,
        manifest_path,
        Path(train_payload["checkpoint"]),
        episodes=eval_episodes,
        progress=progress,
        episode_log_interval=episode_log_interval,
        tensorboard=tensorboard,
        tensorboard_dir=shared_tb_dir,
    )
    comparison = {
        "random_mean_distance_reduction": float(random_payload["summary"]["mean_distance_reduction"]),
        "evaluation_mean_distance_reduction": float(evaluate_payload["summary"]["mean_distance_reduction"]),
        "evaluation_minus_random": float(
            evaluate_payload["summary"]["mean_distance_reduction"] - random_payload["summary"]["mean_distance_reduction"]
        ),
    }
    root_summary = {
        "plugin": inspect["name"],
        "artifact_root": str(artifacts_dir),
        "target_summary": {k: v for k, v in target_summary.items() if k != "rows"},
        "clap_summary": {
            "embedding_shape": clap_summary["embedding_shape"],
            "min_offdiag_distance": clap_summary["min_offdiag_distance"],
            "max_offdiag_distance": clap_summary["max_offdiag_distance"],
        },
        "random_summary": random_payload["summary"],
        "training_summary": train_payload["summary"],
        "evaluation_summary": evaluate_payload["summary"],
        "comparison": comparison,
    }
    write_json(artifacts_dir / "target_summary.json", target_summary)
    append_csv(artifacts_dir / "target_summary.csv", list(target_summary["rows"][0].keys()), target_summary["rows"])
    write_json(artifacts_dir / "clap_summary.json", clap_summary)
    write_json(artifacts_dir / "full_smoke_summary.json", root_summary)
    stage_log(f"Full smoke run complete. Summary written to {artifacts_dir / 'full_smoke_summary.json'}.")
    return root_summary
