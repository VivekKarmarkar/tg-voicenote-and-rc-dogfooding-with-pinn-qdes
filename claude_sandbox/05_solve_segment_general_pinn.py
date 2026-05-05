import sys
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random, jit, vmap, jacfwd
import optax
import matplotlib.pyplot as plt
from pathlib import Path
import time as time_module

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FF_SCALE = 4.0
MAX_NUM_FF = 259
HIDDEN = 128
BUDGET = 280.0


def compute_num_ff(dt_real):
    target_ff = int(np.ceil(2.0 * np.pi * 10.0 * dt_real / FF_SCALE))
    return min(max(64, target_ff), MAX_NUM_FF)


def load_data_csv(csv_path):
    df = pd.read_csv(csv_path)
    t = jnp.array(df["time"].values)
    gyr = jnp.stack([jnp.array(df["Gyr_X"].values), jnp.array(df["Gyr_Y"].values), jnp.array(df["Gyr_Z"].values)], axis=-1)
    return t, gyr


def load_metadata_csv(csv_path):
    df = pd.read_csv(csv_path)
    row = df.iloc[0]
    qi = jnp.array([row["q_init_w"], row["q_init_x"], row["q_init_y"], row["q_init_z"]])
    qf = jnp.array([row["q_final_w"], row["q_final_x"], row["q_final_y"], row["q_final_z"]])
    return qi, qf


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


def fourier_features(t_scalar, num_ff):
    freqs = jnp.arange(1, num_ff + 1) * FF_SCALE
    t_expanded = t_scalar * freqs
    return jnp.concatenate([jnp.sin(t_expanded), jnp.cos(t_expanded), jnp.array([t_scalar])])


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


def predict_q_scalar(params, t_scalar, num_ff):
    ff = fourier_features(t_scalar, num_ff)
    direct = ff @ params[0][0] + params[0][1]
    correction = mlp_forward(params[1:], ff)
    raw = direct + correction
    return normalize_quat(raw)


def predict_q_batch(params, t_norm, num_ff):
    return vmap(predict_q_scalar, in_axes=(None, 0, None))(params, t_norm, num_ff)


def compute_omega_single(params, t_scalar, dt_real, num_ff):
    q = predict_q_scalar(params, t_scalar, num_ff)
    dq_dt_norm = jacfwd(predict_q_scalar, argnums=1)(params, t_scalar, num_ff)
    dq_dt_real = dq_dt_norm / dt_real
    q_conj = quat_conjugate(q)
    omega_quat = 2.0 * quat_multiply(q_conj, dq_dt_real)
    return omega_quat[1:]


def compute_omega_batch(params, t_norm, dt_real, num_ff):
    return vmap(compute_omega_single, in_axes=(None, 0, None, None))(params, t_norm, dt_real, num_ff)


def quat_distance(q1, q2):
    dot = jnp.sum(q1 * q2, axis=-1)
    return 1.0 - jnp.abs(dot)


def loss_fn(params, t_norm, gyr, dt_real, num_ff, qi, qf, w_bc, w_omega, w_max):
    q_pred = predict_q_batch(params, t_norm, num_ff)
    loss_bc = quat_distance(q_pred[0:1], qi[None])[0] + quat_distance(q_pred[-1:], qf[None])[0]
    omega_pred = compute_omega_batch(params, t_norm, dt_real, num_ff)
    omega_err = omega_pred - gyr
    loss_omega = jnp.mean(jnp.sum(omega_err**2, axis=-1))
    omega_err_abs = jnp.abs(omega_err)
    loss_max = jnp.mean(jnp.maximum(omega_err_abs - 0.05, 0.0) ** 2)
    max_omega_err = jnp.max(omega_err_abs)
    total = w_bc * loss_bc + w_omega * loss_omega + w_max * loss_max
    return total, (loss_bc, loss_omega, max_omega_err)


def make_chunked_plot(t_np, gyr_np, omega_pred_np, passed, seg_name, dt_real):
    chunk_size = 35.0
    n_chunks = int(np.ceil(dt_real / chunk_size))
    n_cols = 3
    n_rows = int(np.ceil(n_chunks / n_cols))
    ENVELOPE = 0.1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), squeeze=False)
    labels = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
    colors_meas = ['#1f77b4', '#2ca02c', '#d62728']
    colors_pred = ['#ff7f0e', '#9467bd', '#8c564b']

    t_start = t_np[0]
    for chunk_idx in range(n_chunks):
        row, col = chunk_idx // n_cols, chunk_idx % n_cols
        ax = axes[row, col]
        t_lo = t_start + chunk_idx * chunk_size
        t_hi = t_start + (chunk_idx + 1) * chunk_size
        mask = (t_np >= t_lo) & (t_np < t_hi)
        t_c = t_np[mask]
        gyr_c = gyr_np[mask]
        pred_c = omega_pred_np[mask]
        for i in range(3):
            ax.plot(t_c, gyr_c[:, i], '-', color=colors_meas[i], linewidth=0.8, alpha=0.7)
            ax.fill_between(t_c, gyr_c[:, i] - ENVELOPE, gyr_c[:, i] + ENVELOPE,
                            color=colors_meas[i], alpha=0.1)
            ax.plot(t_c, pred_c[:, i], '--', color=colors_pred[i], linewidth=1.5)
        ax.set_title(f'{t_lo:.0f}s – {t_hi:.0f}s')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)

    for chunk_idx in range(n_chunks, n_rows * n_cols):
        row, col = chunk_idx // n_cols, chunk_idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f'{seg_name} — Chunked View (Pass: {"YES" if passed else "NO"})', fontsize=14)
    plt.tight_layout()
    out_path = RESULTS_DIR / f"{seg_name}_pinn_omega_chunked.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <data.csv> <metadata.csv>")
        sys.exit(1)

    from gpu_test import run_gpu_check
    run_gpu_check()

    data_csv = sys.argv[1]
    meta_csv = sys.argv[2]

    print(f"Loading data from {data_csv}...")
    t_full, gyr_full = load_data_csv(data_csv)
    QI, QF = load_metadata_csv(meta_csv)

    n_samples = len(t_full)
    t_min, t_max = float(t_full[0]), float(t_full[-1])
    dt_real = t_max - t_min
    t_norm_full = (t_full - t_min) / dt_real
    num_ff = compute_num_ff(dt_real)

    t_train = t_norm_full
    gyr_train = gyr_full
    n_train = len(t_train)

    print(f"  {n_samples} samples, {dt_real:.2f}s")
    print(f"  num_ff={num_ff}, FF_SCALE={FF_SCALE}, hidden={HIDDEN}")
    print(f"  Max physical freq: {num_ff * FF_SCALE / (2 * np.pi * dt_real):.1f} Hz")
    print(f"  Training on {n_train} points")
    print(f"  qi: {np.array(QI)}")
    print(f"  qf: {np.array(QF)}")

    input_dim = num_ff * 2 + 1
    key = random.PRNGKey(42)
    key, k1, k2 = random.split(key, 3)
    w_direct = random.normal(k1, (input_dim, 4)) * jnp.sqrt(2.0 / input_dim)
    b_direct = jnp.zeros(4)
    correction_params = init_mlp(k2, [input_dim, HIDDEN, 4])
    params = [(w_direct, b_direct)] + correction_params

    global_start = time_module.time()

    # Training with cosine decay LR
    print(f"\nTraining (budget: {BUDGET}s, cosine decay 1e-3 → 1e-5)...")
    schedule = optax.cosine_decay_schedule(1e-3, 200000, alpha=0.01)
    opt = optax.adam(schedule)
    opt_state = opt.init(params)

    @jit
    def step(params, opt_state):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, t_train, gyr_train, dt_real, num_ff, QI, QF, 200.0, 2.0, 20.0)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    t0 = time_module.time()
    epoch = 0
    while True:
        params, opt_state, loss, (loss_bc, loss_omega, max_err) = step(params, opt_state)
        elapsed = time_module.time() - t0
        if epoch % 5000 == 0:
            print(f"  Epoch {epoch:6d} | Loss: {float(loss):.4f} | BC: {float(loss_bc):.6f} | "
                  f"Omega: {float(loss_omega):.4f} | MaxErr: {float(max_err):.4f} rad/s | {elapsed:.1f}s/{BUDGET}s")
        if elapsed >= BUDGET:
            print(f"  Training done: {epoch} epochs in {elapsed:.1f}s | MaxErr: {float(max_err):.4f}")
            break
        epoch += 1

    total_time = time_module.time() - global_start
    print(f"\nTotal training time: {total_time:.1f}s")

    # Evaluate on ALL points
    print("\nEvaluating on all points...")
    omega_pred = compute_omega_batch(params, t_norm_full, dt_real, num_ff)
    omega_pred_np = np.array(omega_pred)
    gyr_np = np.array(gyr_full)
    t_np = np.array(t_full)

    abs_err = np.abs(omega_pred_np - gyr_np)
    max_err_per_comp = abs_err.max(axis=0)
    print(f"  Max |error| per component: x={max_err_per_comp[0]:.4f}, y={max_err_per_comp[1]:.4f}, z={max_err_per_comp[2]:.4f} rad/s")
    print(f"  Overall max |error|: {abs_err.max():.4f} rad/s")
    passed = abs_err.max() <= 0.10
    print(f"  Pass (±0.10 rad/s): {'YES' if passed else 'NO'}")

    # Plot 1: full overlay
    ENVELOPE = 0.1
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    labels = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
    colors_meas = ['#1f77b4', '#2ca02c', '#d62728']
    colors_pred = ['#ff7f0e', '#9467bd', '#8c564b']

    for i in range(3):
        ax = axes[i]
        ax.plot(t_np, gyr_np[:, i], '-', color=colors_meas[i], linewidth=1, alpha=0.8, label=f'Measured {labels[i]}')
        ax.fill_between(t_np, gyr_np[:, i] - ENVELOPE, gyr_np[:, i] + ENVELOPE,
                         color=colors_meas[i], alpha=0.15, label=f'±{ENVELOPE} rad/s envelope')
        ax.plot(t_np, omega_pred_np[:, i], '--', color=colors_pred[i], linewidth=2.5,
                label=f'PINN {labels[i]}')
        ax.set_ylabel(f'{labels[i]} (rad/s)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f'PINN (gyro-only BVP) — Angular Velocity Overlay (Pass: {"YES" if passed else "NO"})')
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    seg_name = Path(data_csv).stem
    out_path = RESULTS_DIR / f"{seg_name}_pinn_omega.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")

    # Plot 2: chunked view for large datasets (>100s)
    if dt_real > 60.0:
        chunked_path = make_chunked_plot(t_np, gyr_np, omega_pred_np, passed, seg_name, dt_real)
        print(f"Saved chunked: {chunked_path}")

    print(f"Total wall time (including eval+plot): {time_module.time() - global_start:.1f}s")
