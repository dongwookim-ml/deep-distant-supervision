1. Install cuda in a local directory. Go to home folder and run

`wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run`

 - As of Oct, 2018, Cuda 9.2 cannot be installed in cray because the driver installed does not satisfy the requirement of 9.2
 - You can download the older version of cuda from https://developer.nvidia.com/cuda-toolkit-archive
 - Download configuration was : Linux / x86_64 / Ubuntu / 17.04 / runfile (local)

2. Add execution permission to file
`chmod +x cuda_9.0.176_384.81_linux-run`

3. Install cuda in your local directory:
`./cuda_9.0.176_384.81_linux-run --silent --toolkit --toolkitpath=$HOME/cuda/toolkit --samples --samplespath=$HOME/cuda/samples --override`

4. Add these two lines to your `.bashrc` file
`export PATH=$HOME/cuda/toolkit/bin:$PATH`
`export LD_LIBRARY_PATH=$HOME/cuda/toolkit/lib64/:$LD_LIBRARY_PATH`

5. Update your environment
`source ~/.bashrc`

6. If you are using conda(miniconda, anaconda), create a new environment with 
`conda create -n torch python=3.7`

7. After activating virtual environment
`conda activate torch`

8. Install pytorch using
`conda install pytorch torchvision -c pytorch`

9. Run python and try
`import torch`
`torch.cuda.is_available()`

10. If it returns True, now you can use cuda to run the model.

11. Optionially, you can check the current status of GPUs via `nvidia-smi` command (shell).
