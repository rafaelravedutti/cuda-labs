//#include <wb.h>
#include "/home/rrlmachado/labs/wb4.h" // use our lib instead (under construction)
//#include "/home/wagner/ci853/labs-achel/wb.h" // use our lib instead (under construction)

#include <string.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x < len) {
    out[x] = in1[x] + in2[x];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  int wrongSolution;
  int i;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);
  // show_args( args ); // debug

  wbTime_start(Generic, "Importing data and creating memory on host");

  hostInput1 = (float *)wbImport( wbArg_getInputFile(args, 1), &inputLength );
  hostInput2 = (float *)wbImport( wbArg_getInputFile(args, 2), &inputLength );
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **) &deviceInput1, inputLength * sizeof(float));
  cudaMalloc((void **) &deviceInput2, inputLength * sizeof(float));
  cudaMalloc((void **) &deviceOutput, inputLength * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  dim3 grid((inputLength - 1) / 128 + 1, 1, 1);
  dim3 block(128, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  vecAdd<<<grid, block>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  //wbSolution(args, hostOutput, inputLength);

  wrongSolution = 0;

  for(i = 0; i < inputLength; ++i) {
    if(hostOutput[i] != hostInput1[i] + hostInput2[i]) {
      wrongSolution = 1;
      fprintf(stdout, "Error at position %i (%.4f != %.4f).\n", i, hostOutput[i], hostInput1[i] + hostInput2[i]);
    }
  }

  if(wrongSolution == 0) {
    fprintf(stdout, "Solution is right!\n");
  }

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
