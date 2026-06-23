# Installing JAX on Dawn

## 1. Introduction

This is guidance for installing [JAX](https://docs.jax.dev/en/latest/)
in a [conda](https://docs.conda.io/en/latest/) environment on
the [Dawn supercomputer](https://www.hpc.cam.ac.uk/d-w-n).  Dawn is
hosted at the University of Cambridge, and is part
of the [AI Resource Research (AIRR)](https://www.gov.uk/government/publications/ai-research-resource/airr-advanced-supercomputers-for-the-uk).  It was
initially installed with 256 nodes, in the form of [Dell PowerEdge XE9640](https://www.delltechnologies.com/asset/en-us/products/servers/technical-support/poweredge-xe9640-spec-sheet.pdf) servers.  Each node consisted of: 2 CPUs ([Intel Xeon Platinum 8468](https://www.intel.com/content/www/us/en/products/sku/231735/intel-xeon-platinum-8468-processor-105m-cache-2-10-ghz/specifications.html)), each with 48 cores and 512 GiB RAM; 4 GPUs ([Intel Data Centre GPU Max 1550](https://www.intel.com/content/www/us/en/products/sku/232873/intel-data-center-gpu-max-1550/specifications.html)),
each with two stacks (or tiles), 1024 compute units, and 128 GiB RAM.

The material collected here is licensed under the
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## 2. Installation

In case you don't already have your own `conda` installation, you can find
guidance for installing `conda` on Dawn at:
- [https://github.com/kh296/dawn-conda](https://github.com/kh296/dawn-conda)

Installation of JAX may be performed
[via a Slurm job](#21-installation-via-a-slurm-job) or
[from the command line](#22-installation-from-the-command-line).

### 2.1 Installation via a Slurm job

On a Dawn login node or compute node, clone this repository,
and move to the scripts directory:
```
git clone https://github.com/kh296/dawn-jax
cd dawn-jax/scripts
```

Submit a Slurm job to run the installation script:
```
# Substitute for <project_account> a valid project account.
# Substitute for <partition> a valid partition.
# Set CONDA_INSTALL to the path of your conda installation.
sbatch --account=<project_account> --partition=<partition> --export=CONDA_INSTALL="~/miniforge3" ./jax_install.sh
```

Once it starts running, the script should take about five minutes to
complete.  The job output is written to `jax_install.log`.  If the
installation is successful, the output includes results from minimal
checks that the allocated GPU(s) are visible, and that an array can be
created.  These checks produce warnings, but shouldn't produce errors.  The
last line of the output is the command to set up the environment
for using JAX.  This command references the file `../envs/jax-setup.sh`,
created during installation.

### 2.2 Installation from the command line

On a Dawn compute node, clone this repository, and move to
the scripts directory:
```
git clone https://github.com/kh296/dawn-jax
cd dawn-jax/scripts
```

Run the installation script:
```
# Substitute for <project_account> a valid project account.
# Substitute for <partition> a valid partition.
# Set CONDA_INSTALL to the path of your conda installation.
CONDA_INSTALL="~/miniforge3" ./jax_install.sh |& tee jax_install.log
```

Output is written both to terminal and to the file `jax_install.log`.
If the installation is successful, the output includes results from minimal
checks that the allocated GPU(s) are visible, and that an array can be
created.  These checks produce warnings, but shouldn't produce errors.  The
last line of the output is the command to set up the environment
for using JAX.  This command references the file `../envs/jax-setup.sh`,
created during installation.

## 3. Further information

Installation of `JAX` on Dawn is based on the installation for
[Accelerated JAX on Intel GPU](https://github.com/intel/intel-extension-for-openxla/blob/main/docs/acc_jax.md), with compatible package versions taken
from the [intel-extension-for-openxla PyPI documentation](https://pypi.org/project/intel-extension-for-openxla/).

The installation script [scripts/jax_install.sh](scripts/jax_install.sh)
provides several options, for example allowing installation to a
`conda` environment with a name different from the default (`jax`).  For
more information, from the `scripts` directory run:
```
./jax_install.sh -h
```
