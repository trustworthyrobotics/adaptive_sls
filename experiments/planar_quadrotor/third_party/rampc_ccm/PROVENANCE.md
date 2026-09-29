# Official RAMPC-CCM artifacts

The files in `data/` were copied without modification from the public ETH
Zurich repository:

- repository: `https://gitlab.ethz.ch/ics/RAMPC-CCM.git`
- commit: `fbab02d8acd955e1fc18d40b651d747bbe431367`
- retrieved: 2026-09-01
- license: MIT; see `LICENSE`

`constants.mat` and
`own_rccm_0.7_w_0.1_th_0.01_pd_3.14_sc_-2.5.mat` contain the authors'
precomputed constants and polynomial CCM/differential controller.

The original SOS certificate applies to the uncertainty used in the paper,
not automatically to this experiment's two-axis inertial wind. The Python
experiment therefore loads the official polynomial metric as a candidate and
runs a separate sampled contraction audit for the changed dynamics. A sampled
audit is explicitly not represented as an SOS certificate.

