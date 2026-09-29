# Local MPX snapshot

This directory contains the MPX modules and Go2 assets used by the
`quadruped` parity experiment. They were copied from the `gpu_sls` conda
environment so the adaptive experiment does not import files from the sibling
`gpu_sls` repository or depend on that environment at runtime.

The parity-critical files are byte-for-byte copies of that environment:

- `mpx/utils/models.py`
- `mpx/utils/mpc_utils.py`
- `mpx/utils/objectives.py`
- `mpx/utils/sim.py`
- `mpx/data/go2/go2_mjx.xml`
- `mpx/data/go2/scene_mjx.xml`
- `mpx/data/go2/assets/`

`quadruped.py` prepends this `vendor` directory to `sys.path` before importing
MPX. This keeps the experiment self-contained and prevents the active conda
environment from silently selecting a different MPX revision.
