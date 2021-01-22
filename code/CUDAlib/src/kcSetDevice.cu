#include "mex.h"
#include <cuda_runtime.h>
#include "kcDefs.h" //see for info on anything starting with KC_
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])  {
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    cudaError_t ce;
    
    for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
    {
        ce = cudaSetDevice(deviceIndex);   
        if(ce == cudaSuccess) {
            mexPrintf("Selected CUDA device %d\n", deviceIndex);
            return;
        }else{
            mexPrintf("Error selecting device %d ", deviceIndex);
            mexPrintf(cudaGetErrorString(ce));
            mexPrintf(" (%d)\n", (int)ce);
            mexErrMsgTxt("CUDA Errors");

        }
    }

    mexPrintf("Unable to select any devices");

}
