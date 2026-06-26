# Installing JAX on AMD Accelerator Cluster

JAX can be installed in a `conda` environment on the
[AMD Accelerator Cluster (AAC)](https://aac.amd.com/help/) by following
the instructions for [Installing JAX on Dawn](../README.md),
except that for installation via a Slurm job the submission command needs
to be different.  In particular, it's usually not necessary
to specify the account, but it is necessary to specify the partition,
and the resources.

In case you don't already have your own `conda` installation,
first follow the guidance for [Installing conda on AMD Accelerator Cluster](https://github.com/kh296/dawn-conda/blob/main/docs/aac.md).

On AAC6, after following the guidance of
[Installing JAX on Dawn](../README.md)
for obtaining the JAX installation script, an example submission command is:
```
# Set CONDA_INSTALL to the path of your conda installation.
sbatch --partition=1CN192C4G1H_MI300A_Ubuntu22  --cpus-per-gpu=48 --export=CONDA_INSTALL="~/miniforge3" ./jax_install.sh
```
Installation of JAX on AAC6 using
[scripts/jax_install.sh](../scripts/jax_install.sh)
is based on the documentation for
[pip installation: AMD GPU (ROCm)](https://docs.jax.dev/en/latest/installation.html#pip-installation-amd-gpu-rocm).
