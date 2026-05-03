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


def loss_fn_data_only(params, t_norm, q_ref, q_start, q_end):
    q_pred = predict_q_batch(params, t_norm)
    loss_bc = quat_distance(q_pred[0:1], q_start[None])[0] + quat_distance(q_pred[-1:], q_end[None])[0]
    loss_data = jnp.mean(quat_distance(q_pred, q_ref))
    total = 50.0 * loss_bc + loss_data
    return total, (loss_bc, loss_data)


def loss_fn_full(params, t_norm, t_phys, gyr_phys, q_ref, q_start, q_end, dt_real):
    q_pred = predict_q_batch(params, t_norm)
    loss_bc = quat_distance(q_pred[0:1], q_start[None])[0] + quat_distance(q_pred[-1:], q_end[None])[0]
    loss_data = jnp.mean(quat_distance(q_pred, q_ref))
    omega_pred = compute_omega_vmapped(params, t_phys, dt_real)
    loss_physics = jnp.mean(jnp.sum((omega_pred - gyr_phys)**2, axis=-1))
    total = 50.0 * loss_bc + loss_data + 0.01 * loss_physics
    return total, (loss_bc, loss_physics, loss_data)


def solve_segment(seg_id, pretrain_epochs=3000, physics_epochs=5000, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"Solving segment {seg_id}")
    print(f"{'='*60}")

    t, gyr, q_ref, q_start, q_end = load_segment(seg_id)
    n_samples = len(t)
    t_min, t_max = float(t[0]), float(t[-1])
    dt_real = t_max - t_min
    t_norm = (t - t_min) / dt_real
    print(f"  Samples: {n_samples}, Duration: {dt_real:.2f}s")

    input_dim = NUM_FF * 2 + 1
    key = random.PRNGKey(seg_id)
    layer_sizes = [input_dim, 256, 256, 128, 4]
    params = init_mlp(key, layer_sizes)

    # Phase 1: data-only
    print("  Phase 1: Data-only pretraining...")
    opt1 = optax.adam(optax.cosine_decay_schedule(lr, pretrain_epochs))
    opt_state1 = opt1.init(params)

    @jit
    def step_data(params, opt_state):
        (loss, aux), grads = jax.value_and_grad(loss_fn_data_only, has_aux=True)(
            params, t_norm, q_ref, q_start, q_end
        )
        updates, opt_state = opt1.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    t0 = time_module.time()
    for epoch in range(pretrain_epochs):
        params, opt_state1, loss, (loss_bc, loss_data) = step_data(params, opt_state1)
        if epoch % 1000 == 0 or epoch == pretrain_epochs - 1:
            print(f"    Epoch {epoch:5d} | Loss: {float(loss):.6f} | BC: {float(loss_bc):.6f} | "
                  f"Data: {float(loss_data):.6f} | {time_module.time()-t0:.1f}s")

    # Phase 2: add physics
    print("  Phase 2: Physics-informed training...")
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
            params, t_norm, t_phys, gyr_phys, q_ref, q_start, q_end, dt_jnp
        )
        updates, opt_state = opt2.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    t0 = time_module.time()
    for epoch in range(physics_epochs):
        params, opt_state2, loss, (loss_bc, loss_physics, loss_data) = step_full(params, opt_state2)
        if epoch % 1000 == 0 or epoch == physics_epochs - 1:
            print(f"    Epoch {epoch:5d} | Loss: {float(loss):.6f} | BC: {float(loss_bc):.6f} | "
                  f"Phys: {float(loss_physics):.6f} | Data: {float(loss_data):.6f} | {time_module.time()-t0:.1f}s")

    # Evaluate
    q_pred = predict_q_batch(params, t_norm)
    omega_pred = compute_omega_vmapped(params, t_norm[phys_idx], dt_jnp)

    t_np = np.array(t)
    q_pred_np = np.array(q_pred)
    q_ref_np = np.array(q_ref)
    gyr_np = np.array(gyr)
    t_eval_np = np.array(t[phys_idx])
    omega_pred_np = np.array(omega_pred)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for i, lbl in enumerate(['w', 'x', 'y', 'z']):
        axes[0].plot(t_np, q_ref_np[:, i], '--', label=f'ref q_{lbl}', alpha=0.7, linewidth=1.5)
        axes[0].plot(t_np, q_pred_np[:, i], '-', label=f'pred q_{lbl}', alpha=0.7, linewidth=1)
    axes[0].set_title(f'Segment {seg_id}: Quaternion components')
    axes[0].legend(fontsize=7, ncol=4)

    for i, lbl in enumerate(['x', 'y', 'z']):
        axes[1].plot(t_np, gyr_np[:, i], '--', alpha=0.4, linewidth=0.8, label=f'meas ω_{lbl}')
        axes[1].plot(t_eval_np, omega_pred_np[:, i], '-', alpha=0.7, linewidth=1, label=f'pred ω_{lbl}')
    axes[1].set_title(f'Segment {seg_id}: Angular velocity')
    axes[1].legend(fontsize=7, ncol=3)

    dot = np.sum(q_pred_np * q_ref_np, axis=1)
    q_error = 1.0 - np.abs(dot)
    axes[2].plot(t_np, q_error, 'r-')
    axes[2].set_title(f'Segment {seg_id}: Quaternion distance')
    axes[2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"segment_{seg_id:02d}_result.png", dpi=150)
    plt.close()

    mean_err = float(np.mean(q_error))
    max_err = float(np.max(q_error))
    print(f"  Mean q dist: {mean_err:.6f}, Max: {max_err:.6f}")
    return {"segment": seg_id, "n_samples": n_samples, "duration": dt_real,
            "mean_q_dist": mean_err, "max_q_dist": max_err}


if __name__ == "__main__":
    test_segments = [11, 23, 4]
    results = []
    for sid in test_segments:
        r = solve_segment(sid, pretrain_epochs=5000, physics_epochs=8000)
        results.append(r)

    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    for r in results:
        print(f"  Seg {r['segment']:2d} | {r['n_samples']:4d} pts | {r['duration']:.2f}s | "
              f"Mean: {r['mean_q_dist']:.6f} | Max: {r['max_q_dist']:.6f}")
