Code to run the stepping/ramping model comparison described in
    Kenneth W. Latimer, Jacob L. Yates, Miriam L. R. Meister, Alexander C. Huk, 
    & Jonathan W. Pillow (2015). Single-trial spike trains in parietal cortex 
    reveal discrete steps during decision-making. Science, 349(6244):184-187.
This code takes in spike data observed over many trials, fits both a "stepping"
and "ramping" model to the data, and compares the model fits with the Deviance
Information Criterion (DIC).

Requirements:
    MATLAB version R2013a (or above) with statistics toolbox
    Nvidia CUDA toolkit (version 7.0 recommended)
    Nvidia graphics card (compute version 1.3 or above. 
        At least 1GB of graphics memory is recommended.
        Apologies to any AMD users.)
Recommended:
    Ubuntu 15.04
    16+GB RAM

The tools provided run in MATLAB and use Nvidia's CUDA toolkit to accelerate
compution. Running this package requires an Nvidia GPU.
This code has been tested with MATLAB R2013a with CUDA version 7.0 running on
Ubuntu 15.04. The graphics card used was an Nvidia Titan Z.

Previous versions of this package have been tested on Windows 7 and Mac OSX.
However, this release is targeted towards Ubuntu users.
Compiling the CUDA tools on OSX works similarly to the compilation on Ubuntu.
If running this code on Windows is absolutely necessary, the author may provide
limited support in getting the GPU code functional on windows.

To complete the CUDA setup, you must include the following lines to the end of your
.bashrc file in your home directory:
    export PATH=/usr/local/cuda-7.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
(make sure the directories match your CUDA install directory!)

To compile the CUDA files into MEX files that can be executed by MATLAB, see
    code/CUDAlib/myPaths.m (tells the compiler where to find your MATLAB and
                            CUDA installations)
    code/CUDAlib/compileAllMexFiles.m (runs the compile/link scripts)
    
WARNING: Using these CUDA tools and MATLAB's native GPU functions  in the same
         session may cause a library conflict that will crash MATLAB.

Known problems setting up the CUDA/MEX files:
    Compiler error: "mex.h" not found
    Solution: this probably means your MATLAB directory is not set up properly
              in code/CUDAlib/myPath.m
    
    Linking error: undefined reference to `__cxa_atexit'
    Solution: append "-lm -lstdc++" to the CXXLIBS option in your mexopts.sh file

Running the model comparison:
The script exampleScript.m simulates data from one of the two models and runs
the model comparison function. This script calls
    code/runModelComparison.m.
This function takes the data in the form of a 'timeSeries' object described in
the function header. The timeSeries is a processed form of the spike data
observed over all trials. Only the window of interest in each trial is included.
(i.e., spikes before integration and after integration have already been pruned)

For examples of what the 'timeSeries' structure should look like, see the 
functions
    simulateRampingModel.m
    simulateSteppingModel.m
These functions generate a simulated timeSeries object with all the necessary
parts.

All settings, including how many MCMC samples to get, are given in
    setupMCMCParams.m



