import jax
import jax.numpy as jnp
import time


def run_gpu_check():
    print("=" * 50)
    print("GPU CHECK — matmul speed test")
    print("=" * 50)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    size = 4096
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (size, size))
    b = jax.random.normal(key, (size, size))

    c = jnp.dot(a, b).block_until_ready()

    t0 = time.time()
    for _ in range(10):
        c = jnp.dot(a, b).block_until_ready()
    elapsed = time.time() - t0

    print(f"{size}x{size} matmul, 10 iterations: {elapsed:.3f}s ({elapsed/10*1000:.1f}ms each)")
    print(f"Device used: {c.devices()}")
    print("=" * 50)
    print()


if __name__ == "__main__":
    run_gpu_check()
