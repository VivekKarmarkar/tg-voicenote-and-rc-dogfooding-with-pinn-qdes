import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import optax
import matplotlib.pyplot as plt
from pathlib import Path
import time as time_module

SEGMENTS_DIR = Path(__file__).parent / "segments"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

NUM_FF = 48
FF_SCALE = 3.0


def load_segment(seg_id):
    with open(SEGMENTS_DIR / f"segment_{seg_id:02d}.json") as f:
        data = json.load(f)
    t = jnp.array(data["time"])
    gyr = jnp.stack([jnp.array(data["gyr_x"]), jnp.array(data["gyr_y"]), jnp.array(data["gyr_z"])], axis=-1)
    q_ref = jnp.stack([jnp.array(data["q_w"]), jnp.array(data["q_x"]), jnp.array(data["q_y"]), jnp.array(data["q_z"])], axis=-1)
    q_start = jnp.array(data["q_start"])
    q_end = jnp.array(data["q_end"])
    return t, gyr, q_ref, q_start, q_end


def quat_conjugate(q):
    return jnp.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_multiply(p, q):
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ], axis=-1)


def normalize_quat(q):
    return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


def fourier_features(t):
    freqs = jnp.arange(1, NUM_FF + 1) * FF_SCALE
    t_expanded = t[..., None] * freqs[None, :]
    t_col = t[..., None] if t.ndim == 1 else t
    return jnp.concatenate([jnp.sin(t_expanded), jnp.cos(t_expanded), t_col], axis=-1)


def init_mlp(key, layer_sizes):
    params = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = random.split(key)
        w = random.normal(subkey, (layer_sizes[i], layer_sizes[i+1])) * jnp.sqrt(2.0 / layer_sizes[i])
        b = jnp.zeros(layer_sizes[i+1])
        params.append((w, b))
    return params


def mlp_forward(params, x):
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


def predict_q_batch(params, t_norm):
    ff = fourier_features(t_norm)
    raw = mlp_forward(params, ff)
    return normalize_quat(raw)


def predict_q_scalar(params, t_scalar):
    t_arr = jnp.atleast_1d(t_scalar)
    return predict_q_batch(params, t_arr)[0]


def compute_omega_single(params, t_scalar, dt_real):
    q = predict_q_scalar(params, t_scalar)
    dq_dt_norm = jax.jacfwd(lambda s: predict_q_scalar(params, s))(t_scalar)
    dq_dt_real = dq_dt_norm / dt_real
    q_conj = quat_conjugate(q)
    omega_quat = 2.0 * quat_multiply(q_conj, dq_dt_real)
    return omega_quat[1:]


compute_omega_vmapped = vmap(compute_omega_single, in_axes=(None, 0, None))


def quat_distance(q1, q2):
    dot = jnp.sum(q1 * q2, axis=-1)
    return 1.0 - jnp.abs(dot)


def quat_to_euler(q):
    """Convert quaternion (w,x,y,z) to Euler angles (roll, pitch, yaw) in radians."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def loss_fn_data_only(params, t_norm, q_ref, q_start, q_end):
    q_pred = predict_q_batch(params, t_norm)
    loss_bc = quat_distance(q_pred[0:1], q_start[None])[0] + quat_distance(q_pred[-1:], q_end[None])[0]
    loss_data = jnp.mean(quat_distance(q_pred, q_ref))
    return 50.0 * loss_bc + loss_data, (loss_bc, loss_data)


def loss_fn_full(params, t_norm, t_phys, gyr_phys, q_ref, q_start, q_end, dt_real):
    q_pred = predict_q_batch(params, t_norm)
    loss_bc = quat_distance(q_pred[0:1], q_start[None])[0] + quat_distance(q_pred[-1:], q_end[None])[0]
    loss_data = jnp.mean(quat_distance(q_pred, q_ref))
    omega_pred = compute_omega_vmapped(params, t_phys, dt_real)
    loss_physics = jnp.mean(jnp.sum((omega_pred - gyr_phys)**2, axis=-1))
    return 50.0 * loss_bc + loss_data + 0.01 * loss_physics, (loss_bc, loss_physics, loss_data)


def solve_segment(seg_id, pretrain_epochs=5000, physics_epochs=8000, lr=1e-3):
    t, gyr, q_ref, q_start, q_end = load_segment(seg_id)
    n_samples = len(t)
    t_min, t_max = float(t[0]), float(t[-1])
    dt_real = t_max - t_min
    t_norm = (t - t_min) / dt_real

    input_dim = NUM_FF * 2 + 1
    key = random.PRNGKey(seg_id)
    params = init_mlp(key, [input_dim, 256, 256, 128, 4])

    # Phase 1
    opt1 = optax.adam(optax.cosine_decay_schedule(lr, pretrain_epochs))
    opt_state1 = opt1.init(params)

    @jit
    def step_data(params, opt_state):
        (loss, aux), grads = jax.value_and_grad(loss_fn_data_only, has_aux=True)(params, t_norm, q_ref, q_start, q_end)
        updates, opt_state = opt1.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    for epoch in range(pretrain_epochs):
        params, opt_state1, loss, _ = step_data(params, opt_state1)

    # Phase 2
    n_phys = min(n_samples, 200)
    phys_idx = jnp.linspace(0, n_samples - 1, n_phys).astype(int)
    t_phys = t_norm[phys_idx]
    gyr_phys = gyr[phys_idx]
    dt_jnp = jnp.array(dt_real)

    opt2 = optax.adam(optax.cosine_decay_schedule(lr * 0.1, physics_epochs))
    opt_state2 = opt2.init(params)

    @jit
    def step_full(params, opt_state):
        (loss, aux), grads = jax.value_and_grad(loss_fn_full, has_aux=True)(
            params, t_norm, t_phys, gyr_phys, q_ref, q_start, q_end, dt_jnp)
        updates, opt_state = opt2.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    for epoch in range(physics_epochs):
        params, opt_state2, loss, (loss_bc, loss_phys, loss_data) = step_full(params, opt_state2)

    # Evaluate
    q_pred = predict_q_batch(params, t_norm)
    q_pred_np = np.array(q_pred)
    q_ref_np = np.array(q_ref)
    t_np = np.array(t)

    dot = np.sum(q_pred_np * q_ref_np, axis=1)
    q_error = 1.0 - np.abs(dot)
    mean_err = float(np.mean(q_error))
    max_err = float(np.max(q_error))

    # Euler angles
    roll_pred, pitch_pred, yaw_pred = quat_to_euler(q_pred_np)
    roll_ref, pitch_ref, yaw_ref = quat_to_euler(q_ref_np)

    return {
        "segment": seg_id, "n_samples": n_samples, "duration": dt_real,
        "mean_q_dist": mean_err, "max_q_dist": max_err,
        "final_loss": float(loss), "final_phys": float(loss_phys),
        "t": t_np, "q_pred": q_pred_np, "q_ref": q_ref_np,
        "roll_pred": roll_pred, "pitch_pred": pitch_pred, "yaw_pred": yaw_pred,
        "roll_ref": roll_ref, "pitch_ref": pitch_ref, "yaw_ref": yaw_ref,
        "q_error": q_error,
    }


if __name__ == "__main__":
    n_segments = len(list(SEGMENTS_DIR.glob("segment_*.json")))
    print(f"Running PINN on all {n_segments} segments...")

    all_results = []
    t_total = time_module.time()

    for seg_id in range(n_segments):
        t0 = time_module.time()
        r = solve_segment(seg_id)
        elapsed = time_module.time() - t0
        print(f"  Seg {seg_id:2d} | {r['n_samples']:4d} pts | {r['duration']:.2f}s | "
              f"Mean: {r['mean_q_dist']:.6f} | Max: {r['max_q_dist']:.6f} | {elapsed:.1f}s")
        all_results.append(r)

    total_time = time_module.time() - t_total
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Summary plot: Euler angles for all segments
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    angle_names = ['Roll (Inclination X)', 'Pitch (Inclination Y)', 'Yaw (Heading)']

    for r in all_results:
        t = r["t"]
        for i, (pred, ref, name) in enumerate(zip(
            [r["roll_pred"], r["pitch_pred"], r["yaw_pred"]],
            [r["roll_ref"], r["pitch_ref"], r["yaw_ref"]],
            angle_names
        )):
            axes[i].plot(t, np.degrees(ref), 'b-', alpha=0.3, linewidth=0.5)
            axes[i].plot(t, np.degrees(pred), 'r-', alpha=0.5, linewidth=0.5)

    for i, name in enumerate(angle_names):
        axes[i].set_ylabel(f'{name} (°)')
        if i == 0:
            axes[i].set_title('PINN Euler Angle Estimates — All Segments (blue=ref, red=pred)')
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "all_segments_euler.png", dpi=150)
    plt.close()

    # Summary plot: quaternion error across all segments
    fig, ax = plt.subplots(figsize=(16, 4))
    for r in all_results:
        ax.plot(r["t"], r["q_error"], 'r-', alpha=0.5, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Quaternion distance (1-|dot|)')
    ax.set_title('Quaternion Error Across All Segments')
    ax.set_ylim(0, 0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "all_segments_error.png", dpi=150)
    plt.close()

    # Summary table
    print("\n" + "="*80)
    print(f"{'Seg':>4} | {'Pts':>5} | {'Dur(s)':>6} | {'Mean dist':>10} | {'Max dist':>10} | {'Physics':>10}")
    print("-"*80)
    for r in all_results:
        print(f"{r['segment']:4d} | {r['n_samples']:5d} | {r['duration']:6.2f} | "
              f"{r['mean_q_dist']:10.6f} | {r['max_q_dist']:10.6f} | {r['final_phys']:10.4f}")

    means = [r['mean_q_dist'] for r in all_results]
    maxes = [r['max_q_dist'] for r in all_results]
    print("-"*80)
    print(f"{'AVG':>4} |       |        | {np.mean(means):10.6f} | {np.mean(maxes):10.6f} |")
    print(f"Saved: all_segments_euler.png, all_segments_error.png")
