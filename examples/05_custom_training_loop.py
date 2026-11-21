"""Custom Optax + Flax NNX training loop paired with a Jax Forge model."""

import os

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jaxforge as jf
from jaxforge.contrib import (
    count_opt_steps,
    count_parameters,
    create_cosine_schedule,
    create_tracker,
)


class FallbackMLP(nnx.Module):
    """Deterministic local fallback used when no LLM API key is set in the environment."""

    def __init__(self, rngs: nnx.Rngs) -> None:
        self.fc1 = nnx.Linear(16, 32, rngs=rngs)
        self.fc2 = nnx.Linear(32, 4, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.fc2(nnx.relu(self.fc1(x)))


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        jf.config.set({"model_name": "gemini-3.1-pro", "api_key": api_key})
        model = jf.build(
            "Build a 2-layer Flax NNX MLP mapping 16 float features to 4 output classes."
        )
    else:
        model = FallbackMLP(nnx.Rngs(0))

    print(f"Model parameter count: {count_parameters(model)}")

    rng = np.random.default_rng(42)
    x_train = jnp.asarray(rng.standard_normal((128, 16), dtype=np.float32))
    y_train = jnp.asarray(rng.integers(0, 4, size=(128,), dtype=np.int32))

    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    schedule = create_cosine_schedule(peak_lr=1e-2, warmup_steps=5, total_steps=30)
    tx = optax.adamw(learning_rate=schedule)
    opt_state = tx.init(params)
    tracker = create_tracker("memory")

    @jax.jit
    def train_step(
        cur_params: nnx.State,
        cur_opt_state: optax.OptState,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
    ) -> tuple[nnx.State, optax.OptState, jnp.ndarray]:
        def loss_fn(p: nnx.State) -> jnp.ndarray:
            m = nnx.merge(graphdef, p, rest)
            logits = m(x_batch)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y_batch))

        loss, grads = jax.value_and_grad(loss_fn)(cur_params)
        updates, next_opt_state = tx.update(grads, cur_opt_state, cur_params)
        next_params = optax.apply_updates(cur_params, updates)
        return next_params, next_opt_state, loss

    for step in range(1, 31):
        params, opt_state, loss_val = train_step(params, opt_state, x_train, y_train)
        loss = float(loss_val)
        tracker.log_metrics({"loss": loss}, step=step)
        if step in (1, 15, 30):
            print(f"Step {step:02d} | loss={loss:.4f}")

    model = nnx.merge(graphdef, params, rest)
    print(f"Completed {count_opt_steps(opt_state)} optimizer steps.")


if __name__ == "__main__":
    main()
