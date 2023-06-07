#!/bin/bash
#
# Run everything and copy outputs, figures and notebook to the archive folder.
#
# Two command line parameters are required a unique identifier used as filename (no spaces allowed)
# and a message, similar to a Git commit message:
#
#    ./run.sh "describe_this_version" "Describe in long words what happened here"

set -e

name="$1"
message="$2"
archive_folder="archive/$(date "+%F_%H-%M_%Z")_$name"


# Pre-flight checks
# =================

# Is the Git repository clean?
if [ -n "$(git status --untracked-files=no --porcelain)" ]; then
    echo "There are uncommitted changes in your Git repository. First commit, then run."
    exit 1
fi

# Is the conda environment activated?
if [ -z "${CONDA_DEFAULT_ENV}" ]; then
    # we could also explicitly check for the right conda environment
    echo "No conda environment active, sure you want to continue?"
    exit 2
fi

# Is $name something which can be used for folder names?
if !(echo "${name}" | grep -q -E '^[a-zA-Z]+[0-9a-zA-Z_-]*$'); then
    echo "invalid characters in name='${name}': must not contain spaces, only alphanumeric "
    echo "and - or _ and start with a letter"
    exit 3
fi

if [ -e "${archive_folder}" ]; then
    echo "error: folder '${archive_folder}' already exists"
    exit 4
fi

if git rev-parse "${name}^{tag}" > /dev/null 2>&1; then
    echo "error: Git tag '${name}' already exists"
    exit 5
fi


# Run computations and notebooks
# ==============================

echo "Run it..."

dvc repro lint
dvc repro unit_test
dvc repro figures
dvc repro data_values

# Run notebooks and store output
echo "DISABLED Exectue notebooks..."

# does not run all notebooks via some weird file name heuristic to exclude Untitled.ipynb etc
# note: need to ignore the kernel set in the notebook, because nbconvert does not use
# nb_conda_kernels to find conda environments (and they might not be explicitly created)
# https://github.com/jupyter/nbconvert/issues/515#issuecomment-273582705
# See also: https://stackoverflow.com/a/58068850/859591
#PYTHONPATH=${PYTHONPATH}:${PWD} jupyter nbconvert\
#    --to notebook\
#    --inplace\
#    --ExecutePreprocessor.kernel_name=python3\
#    --execute notebooks/*_*.ipynb > /dev/null


# Archive results
# ===============

echo "Archive results..."

mkdir -p ${archive_folder}/data

cp -aL notebooks ${archive_folder}
cp -aL data/output ${archive_folder}/data
cp -aL data/figures ${archive_folder}/data
cp -aL data/logfile.log ${archive_folder}/data
cp -aL htmlcov ${archive_folder}

# Store timestamps of (input) data to be able to reproduce what was used
ls -Rl data > ${archive_folder}/data_folder_timestamps

# Store Git commit which was used to produce results
echo "${message}" > ${archive_folder}/message

echo "Create Git tag..."
git tag ${name} -m "${message}"
git show -s > ${archive_folder}/git-commit

# Just in case Conda env is not up-to-date, that makes it traceable at least...
conda env export > ${archive_folder}/env.yml

echo "Done!"
