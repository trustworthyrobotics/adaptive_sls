### Installation

The new path-based Hessian approach relies on `immrax`, which support `jax~=0.6.1`. You can install it by 
```bash
pip install immrax[cuda]
```

If you are using newer versions of `jax` like `0.8.1`, you can install an custom version of `immrax` by
```bash
pip install "immrax[cuda] @ git+https://github.com/keyis2/immrax@upgrade_jax"
pip install linrax --no-deps
```