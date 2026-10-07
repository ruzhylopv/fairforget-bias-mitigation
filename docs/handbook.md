# Fairforget-Unlearning Code Utilisation Handbook
_In case you need to run something on the Jupiter cluster_

 Do not run GPU-heavy experiments on the login node. Request a compute allocation with Slurm first. 
 
 This is an instruction how to do it.
## Setting up the environment
Navigate to the `fairforget-unlearning` directory
```bash
#bash

cd /e/project1/e-dev-2026d09-075/fairforget-bias-mitigation
```

If you want to see the working directory in VSCode:
```bash
code .
```
---
### Selecting Interpreter
Ctrl+Shift+P → Python: Select Interpreter

Look for something like:

Python 3.13.5 ('.venv': uv)

> **Important:** This needs to be run once per device

---
### Getting necessary modules
```bash
module load Python/3.13.5
module load PyTorch/2.9.1
module load uv  
```
> **Important:** This needs to be run on each Jupiter instance
---
### Allocating resources
An **example** for allocating a gpu resource on a booster partition:
```bash
salloc --account=e-dev-2026d09-075 --partition=booster --gpus=1 --time=00:10:00 --gres=gpu:1
```
> **Note:** Your job id is `salloc: Pending job allocation <JOB_ID>`


Further slurm documentation see https://slurm.schedmd.com/quickstart.html, or take a look at the cheatsheet.
### Jupyter server
Having the acquired GPU resource, VSCode interface needs to access it somehow, Jupyter server becomes the bridge between the "Run Cell" button and the allocated resource

**1.** Run:
```
uv run jupyter lab --no-browser
``` 
^ This hosts a local server on your end to the configured Python and libraries you just loaded from the available Jupiter modules.

The terminal outputs several links, you can copy 

`http://localhost:8889/lab?token=<YOUR_TOKEN>`

or 

`http://127.0.0.1:8889/lab?token=<YOUR_TOKEN>`


**2.** Go to the notebook you want to run, click "Select Kernel" →"Select Another Kernel" → "Existing Jupyter Server" 

**3. Paste in** the URL and press Enter. The interface will prompt you to name the Server, feel free to enter any value. 

After that you should see your Server as the **selected Kernel**, it is live until you kill the terminal you initialized it in.

These actions should allow you to run .ipynb cells. Have fun

## Exit procedure
If you are done executing code on your allocated resource and still have available time, you can exit just by entering this command
```
exit
``` 
To shut down the Jupyter server, just kill the process by pressing `Ctrl + C`.

## Sanity checks
```bash
# Check if Python is mounted
python --version
# Check if PyTorch is mounted
uv run python -c "import torch; print(torch.__version__)"
# Display active allocations by user
squeue -u $USER
# Display the information about GPU drivers 
# (NVIDIA System Management Interface)
nvidia-smi
# Check job ID
echo $SLURM_JOB_ID
```