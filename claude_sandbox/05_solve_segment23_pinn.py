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

SEG_ID = 23
NUM_FF = 64
FF_SCALE = 4.0

QI = jnp.array([0.0509361714166126, -0.555871643580193, -0.3393518394925133, 0.7571344341250899])
QF = jnp.array([-0.20157458208707, -0.7378551037198243, -0.254136757986071, 0.5919054333348864])


def load_segment(seg_id):
    with open(SEGMENTS_DIR / f"segment_{seg_id:02d}.json") as f:
        data = json.load(f)
    t = jnp.array(data["time"])
    gyr = jnp.stack([jnp.array(data["gyr_x"]), jnp.array(data["gyr_y"]), jnp.array(data["gyr_z"])], axis=-1)
    return t, gyr


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


compute_omega_batch = vmap(compute_omega_single, in_axes=(None, 0, None))


def quat_distance(q1, q2):
    dot = jnp.sum(q1 * q2, axis=-1)
    return 1.0 - jnp.abs(dot)


def loss_fn(params, t_norm, gyr, dt_real, qi, qf, w_bc, w_omega, w_max):
    q_pred = predict_q_batch(params, t_norm)
    loss_bc = quat_distance(q_pred[0:1], qi[None])[0] + quat_distance(q_pred[-1:], qf[None])[0]
    omega_pred = compute_omega_batch(params, t_norm, dt_real)
    omega_err = omega_pred - gyr
    loss_omega = jnp.mean(jnp.sum(omega_err**2, axis=-1))
    omega_err_abs = jnp.abs(omega_err)
    loss_max = jnp.mean(jnp.maximum(omega_err_abs - 0.03, 0.0) ** 2)
    max_omega_err = jnp.max(omega_err_abs)
    total = w_bc * loss_bc + w_omega * loss_omega + w_max * loss_max
    return total, (loss_bc, loss_omega, max_omega_err)


if __name__ == "__main__":
    print(f"Loading segment {SEG_ID} (gyro only)...")
    t, gyr = load_segment(SEG_ID)
    n_samples = len(t)
    t_min, t_max = float(t[0]), float(t[-1])
    dt_real = t_max - t_min
    t_norm = (t - t_min) / dt_real
    dt_jnp = jnp.array(dt_real)
    print(f"  {n_samples} samples, {dt_real:.2f}s")
    print(f"  qi: {np.array(QI)}")
    print(f"  qf: {np.array(QF)}")

    input_dim = NUM_FF * 2 + 1
    key = random.PRNGKey(42)
    params = init_mlp(key, [input_dim, 512, 512, 256, 4])

    # Phase 1: BC-heavy with physics
    P1_EPOCHS = 15000
    lr1 = 1e-3
    print(f"\nPhase 1: BC-anchored physics training ({P1_EPOCHS} epochs)...")
    opt1 = optax.adam(optax.cosine_decay_schedule(lr1, P1_EPOCHS))
    opt_state1 = opt1.init(params)

    @jit
    def step1(params, opt_state):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, t_norm, gyr, dt_jnp, QI, QF, 500.0, 1.0, 0.0)
        updates, opt_state = opt1.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    t0 = time_module.time()
    for epoch in range(P1_EPOCHS):
        params, opt_state1, loss, (loss_bc, loss_omega, max_err) = step1(params, opt_state1)
        if epoch % 3000 == 0 or epoch == P1_EPOCHS - 1:
            print(f"  Epoch {epoch:5d} | Loss: {float(loss):.4f} | BC: {float(loss_bc):.6f} | "
                  f"Omega: {float(loss_omega):.4f} | MaxErr: {float(max_err):.4f} rad/s | {time_module.time()-t0:.1f}s")

    # Phase 2: balanced physics
    P2_EPOCHS = 30000
    lr2 = 3e-4
    print(f"\nPhase 2: Physics-focused training ({P2_EPOCHS} epochs)...")
    opt2 = optax.adam(optax.cosine_decay_schedule(lr2, P2_EPOCHS))
    opt_state2 = opt2.init(params)

    @jit
    def step2(params, opt_state):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, t_norm, gyr, dt_jnp, QI, QF, 100.0, 2.0, 10.0)
        updates, opt_state = opt2.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    t0 = time_module.time()
    for epoch in range(P2_EPOCHS):
        params, opt_state2, loss, (loss_bc, loss_omega, max_err) = step2(params, opt_state2)
        if epoch % 5000 == 0 or epoch == P2_EPOCHS - 1:
            print(f"  Epoch {epoch:5d} | Loss: {float(loss):.4f} | BC: {float(loss_bc):.6f} | "
                  f"Omega: {float(loss_omega):.4f} | MaxErr: {float(max_err):.4f} rad/s | {time_module.time()-t0:.1f}s")

    # Phase 3: max-error refinement
    P3_EPOCHS = 20000
    lr3 = 1e-4
    print(f"\nPhase 3: Max-error refinement ({P3_EPOCHS} epochs)...")
    opt3 = optax.adam(optax.cosine_decay_schedule(lr3, P3_EPOCHS))
    opt_state3 = opt3.init(params)

    @jit
    def step3(params, opt_state):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, t_norm, gyr, dt_jnp, QI, QF, 100.0, 2.0, 50.0)
        updates, opt_state = opt3.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    t0 = time_module.time()
    for epoch in range(P3_EPOCHS):
        params, opt_state3, loss, (loss_bc, loss_omega, max_err) = step3(params, opt_state3)
        if epoch % 5000 == 0 or epoch == P3_EPOCHS - 1:
            print(f"  Epoch {epoch:5d} | Loss: {float(loss):.4f} | BC: {float(loss_bc):.6f} | "
                  f"Omega: {float(loss_omega):.4f} | MaxErr: {float(max_err):.4f} rad/s | {time_module.time()-t0:.1f}s")

    # Evaluate
    print("\nEvaluating on all points...")
    omega_pred = compute_omega_batch(params, t_norm, dt_jnp)
    omega_pred_np = np.array(omega_pred)
    gyr_np = np.array(gyr)
    t_np = np.array(t)

    abs_err = np.abs(omega_pred_np - gyr_np)
    max_err_per_comp = abs_err.max(axis=0)
    print(f"  Max |error| per component: x={max_err_per_comp[0]:.4f}, y={max_err_per_comp[1]:.4f}, z={max_err_per_comp[2]:.4f} rad/s")
    print(f"  Overall max |error|: {abs_err.max():.4f} rad/s")
    passed = abs_err.max() <= 0.05
    print(f"  Pass (±0.05 rad/s): {'YES' if passed else 'NO'}")

    # Plot
    ENVELOPE = 0.05
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

    axes[0].set_title(f'Segment 23 — PINN (gyro-only BVP) — Angular Velocity Overlay (Pass: {"YES" if passed else "NO"})')
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    out_path = RESULTS_DIR / "segment_23_pinn_omega.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")
