I set up the Gipfelsturm project repo on Clariden and fixed the known hardcoded path/account issues from the course repo.

What I changed locally:
- Cloned swiss-ai/lsaie-ss26-gipfelsturm with the Megatron-LM submodule.
- Fixed alps3.toml so its workdir points to our actual repo location.
- Replaced the old /users/schlag/gipfelsturm path in launch.sh and test-infra.sbatch.
- Replaced #SBATCH --account=infra01 with #SBATCH --account=lsaie-ss26.
- Created my working branch: thomas/profiling-compile.
- Added project/benchmark_log.csv so we record every run, including failed ones.

Before working on your part, please do this on Clariden:

1. Connect to Clariden.
2. Clone our shared fork with submodules.
3. Copy alps3.toml to ~/.edf/alps3.toml.
4. Check that no active script still points to /users/schlag or account=infra01.
5. Create your own branch.

Suggested branches:
- Brandy: brandy/attention-backends
- Jacques: jacques/fp8-memory-fusion

Commands once the shared fork exists:

mkdir -p /users/$USER/gipfelsturm
cd /users/$USER/gipfelsturm
git clone --recurse-submodules <OUR_SHARED_FORK_URL>
cd lsaie-ss26-gipfelsturm
mkdir -p ~/.edf
cp alps3.toml ~/.edf/alps3.toml
grep -RIn "/users/schlag\|account=infra01" . --exclude-dir=.git --exclude-dir=Megatron-LM --exclude="*.bak"

Please do not run long jobs yet. First we need one clean baseline result. Thomas is handling the smoke test, baseline throughput runs, NSYS profiling, and the benchmark log.
