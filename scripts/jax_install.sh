#!/bin/bash
#SBATCH --job-name=jax_install  # create a name for your job
#SBATCH --output=%x.log         # job output file
#SBATCH --partition=pvc9        # cluster partition to be used
#SBATCH --nodes=1               # number of nodes
#SBATCH --gres=gpu:1            # number of allocated gpus per node
#SBATCH --time=00:15:00         # total run time limit (HH:MM:SS)

# Script for installing JAX on the Dawn supercomputer.
#
# This installation relies on the user having a conda installation
# at ${CONDA_HOME}.  If CONDA_HOME is null but CONDA_PREFIX is non-null,
# the former is set to be equal to the latter.  If both CONDA_HOME and
# CONDA_PREFIX are null, CONDA_HOME is set to ${HOME}/miniforge3.  In this
# case, if conda isn't available at ${HOME}/miniforge3 then
# the Miniforge3 flavour of conda will be installed by running
# ./miniforge3_install.sh with default settings.
# For information about the Miniforge3 flavour
# of conda, see: https://conda-forge.org/download/
# For information about ./miniforge3_install.sh, use:
# ./miniforge3_install.sh -h
#
# After installation, if the environment variable CONDA_ENV wasn't set,
# the environment for using JAX can be activated by sourcing the file
# jax-setup.sh, created in the directory ../envs relative to where
# the current script is run.  Otherwise, the file to source is
# ../envs/${CONDA_ENV}-setup.sh
#
# On Dawn, the current script may be run interactively on a compute node
# (not on a login node):
# bash ./jax_install.sh
# or it may be submitted from a login node to the Slurm batch system:
# sbatch --account=<project account> ./jax_install.sh

# Exit at first failure.
set -e

PROJECT_NAME="JAX"
PROJECT_NAME_LC="$(echo ${PROJECT_NAME} | tr [:upper:] [:lower:])"

# Parse command-line options.
usage() {
    echo "usage: jax_install.sh [-h] [-c <conda home>] [-e <conda env>]"
    echo "    Install JAX in a conda environment."
    echo "Options:"
    echo "    -h: Print this help."
    echo "    -c: Use conda installation at <conda home>."
    echo "    -e: Create, and install to, conda environment <conda env>."
    echo "If -c omitted, path to conda installation is first non-empty string from:"
    echo "    \"\${CONDA_HOME}\", \"\${CONDA_PREFIX}\", \"\${HOME}/miniforge3\""
    echo "    If last of these is selected, conda will be installed here"
    echo "    if not already present."
    echo "If -e omitted, the name for the conda environment defaults to \"${PROJECT_NAME_LC}\"."
    echo "Any pre-existing conda environment <conda env> (specified with -e)"
    echo "    or \"${PROJECT_NAME_LC}\" (-e omitted) will be removed."
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h)
            usage
	    exit 0
            ;;
        -c)
            if [[ -n "$2" && "$2" != -* ]]; then
                CONDA_HOME="$2"
		shift 2
            else
                echo "-c must be followed by path to conda installation"
                usage
		exit 1
            fi
            ;;
        -e)
            if [[ -n "$2" && "$2" != -* ]]; then
                CONDA_ENV="$2"
                shift 2
            else
                echo "-e must be followed by name of conda environment"
                exit 1
            fi
            ;;
        -*)
            echo "Unknown option: $1"
            usage
	    exit 1
            ;;
    esac
done

if [[ -z "${CONDA_ENV}" ]]; then
    CONDA_ENV=${PROJECT_NAME_LC}
fi

# Determine system being used.
if [[ "$(hostname)" == "pvc-s"* ]]; then
    SYSTEM="Dawn"
elif [[ "$(hostname)" == *"-pl1"* ]]; then
    SYSTEM="aac6"
elif [[ "${OSTYPE}" == "darwin"* ]]; then
    SYSTEM="macOS"
else
    echo "Installation of ${PROJECT_NAME} for ${OSTYPE} on $(hostname) not handled"
    echo "Exiting: $(date)"
    exit
fi

# Check that conda is available.
if [ -z "${CONDA_HOME}" ]; then
    if [ -z "${CONDA_PREFIX}" ]; then
        CONDA_HOME="${HOME}/miniforge3"
        if ! [ -d "${CONDA_HOME}" ]; then
            ./miniforge3_install.sh
        fi
    else
        CONDA_HOME="${CONDA_PREFIX}"
    fi
fi

# Expand path, without following symbolic links.
CONDA_HOME="${CONDA_HOME/#\~/${HOME}}"
CONDA_HOME=$(cd "$(dirname "${CONDA_HOME}")" && pwd -P)/$(basename "${CONDA_HOME}")

if ! [ -d "${CONDA_HOME}" ]; then
    echo "Conda installation not found at ${CONDA_HOME}"
    echo "Exiting: $(date)"
    exit 2
fi

# Perform installation.
echo "Installation of ${PROJECT_NAME} for ${OSTYPE} on $(hostname) started: $(date)"
T0=${SECONDS}

# Create script for environment setup.
ENVS_DIR=$(realpath ..)/envs
mkdir -p ${ENVS_DIR}
SETUP="${ENVS_DIR}/${CONDA_ENV}-setup.sh"
DAWN_SETUP="/dev/null"
MACOS_SETUP="/dev/null"
if [[ "Dawn" == "${SYSTEM}" ]]; then
    DAWN_SETUP="${SETUP}"
elif [[ "aac6" == "${SYSTEM}" ]]; then
    AAC6_SETUP="${SETUP}"
elif [[ "macOS" == "${SYSTEM}" ]]; then
    MACOS_SETUP="${SETUP}"
fi

rm -rf ${SETUP}
cat <<EOF >${SETUP}
# Setup script for ${CONDA_ENV} on ${SYSTEM}.
# Generated on $(hostname), $(date +"%Y-%m-%d (%a) %H:%M:%S %Z").

EOF

cat <<EOF >>${DAWN_SETUP}
# Load modules.
module purge
module load rhel9/default-dawn
source /usr/local/dawn/software/external/intel-oneapi/2025.2.1/setvars.sh

if [[ -z "${ZE_FLAT_DEVICE_HIERARCHY}" ]]; then
    export ZE_FLAT_DEVICE_HIERARCHY="FLAT"
fi 
EOF

cat <<EOF >>${AAC6_SETUP}
# Load modules.
module purge
module load rocm
module load openmpi

# Set network interface for communication:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname
# Possibilities for listing network interfaces include:
# Linux: ip addr, netstat -i, ifconfig
# MacOS: networksetup -listallhardwarereports, netstat -i, ifconfig
export NCCL_SOCKET_IFNAME="enp129s0"
EOF

cat <<EOF >>${SETUP}

# Initialise conda.
source ${CONDA_HOME}/bin/activate

# Activate environment.
EOF

# Set up installation environment.
source ${SETUP}
conda update -n base -c conda-forge conda -y

# Delete any pre-existing environment.
if [ -d "${CONDA_HOME}/envs/${CONDA_ENV}" ]; then
    rm -rf ${CONDA_HOME}/envs/${CONDA_ENV}
fi

# Create and activate the environment.
if [[ "Dawn" == "${SYSTEM}" ]]; then
    EXTRA_PACKAGES=" libstdcxx-ng"
else
    EXTRA_PACKAGES=""
fi
CMD="conda create -n ${CONDA_ENV} -y python=3.12${EXTRA_PACKAGES}"
echo "${CMD}"
eval "${CMD}"
CMD="conda activate ${CONDA_ENV}"
echo "${CMD}" >> "${SETUP}"
eval "${CMD}"

# Install additional packages.
CMD="python -m pip install --upgrade pip"
echo ""
echo "Ensuring pip up to date:"
echo "${CMD}"
eval "${CMD}"
echo ""
echo "Installing packages:"

# For Intel GPUs, adapt from instructions at:
# https://github.com/intel/intel-extension-for-openxla/blob/0.7.0/docs/acc_jax.md.
# For MacOS, adapt from instructions at:
# https://developer.apple.com/metal/jax/
if [[ "Dawn" == "${SYSTEM}" ]]; then
    CMD="python -m pip install intel-extension-for-openxla==0.7.0"
    echo "${CMD}"
    eval "${CMD}"
elif [[ "MacOS" == "${SYSTEM}" ]]; then
    CMD="python -m pip install jax-metal"
    echo "${CMD}"
    eval "${CMD}"
fi
echo ""
if [[ "aac6" == "${SYSTEM}" ]]; then
    CMD="python -m pip install 'jax[rocm7-local]' flax optax"
else
    CMD="python -m pip install jax==0.5.0 jaxlib==0.5.0 flax==0.10.0 optax==0.2.4"
fi
echo "${CMD}"
eval "${CMD}"

T1=${SECONDS}

# Check installation.
echo ""
echo "Checking installation:"

CMD="python -c 'import jax; print(jax.devices())'"
echo ""
echo "${CMD}"
eval "${CMD}"

CMD="python -c 'import jax; print(jax.numpy.arange(10))'"
echo ""
echo "${CMD}"
eval "${CMD}"
T2=${SECONDS}

echo ""
echo "Installation of ${PROJECT_NAME} for ${OSTYPE} on $(hostname) completed: $(date)"
echo "Time for installation: $((${T1}-${T0})) seconds"
echo "Time for installation checks: $((${T2}-${T1})) seconds"

echo ""
echo "Set up environment for ${PROJECT_NAME} with:"
echo "source ${SETUP}"
