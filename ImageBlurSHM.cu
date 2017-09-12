// v0.2 modified by WZ

//#include <wb.h>
//#include "/home/prof/wagner/ci853/labs/wb4.h" // use our lib instead (under construction)
//#include "/home/wagner/ci853/labs-achel/wb.h" // use our lib instead (under construction)
#include "/home/rrlmachado/labs/wb4.h" // use our lib instead (under construction)

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLUR_SIZE     5

/* Block dimensions */
#define BLOCK_DIM_X   32
#define BLOCK_DIM_Y   32

__global__ void blurImageShm(unsigned char *inputImage, unsigned char *outputImage, int width, int height) {
  /* Shared memory area for image data */
  __shared__ unsigned char
    ds_img[BLOCK_DIM_X + BLUR_SIZE * 2][BLOCK_DIM_Y + BLUR_SIZE * 2];

  /* Shared memory area for mask */
  __shared__ unsigned char
    ds_mask[BLOCK_DIM_X + BLUR_SIZE * 2][BLOCK_DIM_Y + BLUR_SIZE * 2];

  /* Coordinates (x,y) in relation to entire image */
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  /* Registers used for indexes calculation */
  int x_dest, y_dest, idx_src;

  /* Indexes i, j and k */
  int i, j, k;

  /* Evaluated dimensions */
  int dim_x, dim_y;

  /* Horizontal dimension */
  if(width % BLOCK_DIM_X != 0 && blockIdx.x == gridDim.x - 1) {
    dim_x = width % BLOCK_DIM_X;
  } else {
    dim_x = BLOCK_DIM_X;
  }

  /* Vertical dimension */
  if(height % BLOCK_DIM_Y != 0 && blockIdx.y == gridDim.y - 1) {
    dim_y = height % BLOCK_DIM_Y;
  } else {
    dim_y = BLOCK_DIM_Y;
  }

  /* The images have 3 channels (R, G and B), for each channel execute the
     blur procedure */
  for(k = 0; k < 3; ++k) {
    /* Sum is used as accumulator for neighbor pixels */
    int sum = 0;
    /* Counter is the number of valid neighbor pixels to get the average */
    int counter = 0;

    /* Position (x,y) must be inside the image to avoid out of the memory access,
       as thread index can't be negative, only check for width and height */
    if(x < width && y < height) {
      /* At start, copy into the shared memory the region of the image that is
         in the block region, no need for boundary checking in this copy, all
         the image data is copied in the shared memory after the sentinel region */
      ds_mask[threadIdx.x + BLUR_SIZE][threadIdx.y + BLUR_SIZE] = 1;
      ds_img[threadIdx.x + BLUR_SIZE][threadIdx.y + BLUR_SIZE] = \
        inputImage[(y * width + x) * 3 + k];
    }

    /* If the thread x axis is lower than the blur size, use the thread to
       copy the left center and right center regions of the sentinel */
    if(threadIdx.x < BLUR_SIZE) {
      /* The y axis in the shared memory is the same in both regions */
      y_dest = threadIdx.y + BLUR_SIZE;

      /* The x axis for the left region */
      x_dest = threadIdx.x;
      /* The position at the image data is (x-BLUR_SIZE,y) */
      idx_src = (y * width + x - BLUR_SIZE) * 3 + k;

      /* Check if the position is valid and if so, performs the copy, otherwise
         just mark this position as invalid in the mask */
      if(x - BLUR_SIZE > -1) {
        ds_mask[x_dest][y_dest] = 1;
        ds_img[x_dest][y_dest] = inputImage[idx_src];
      } else {
        ds_mask[x_dest][y_dest] = 0;
      }

      /* The x axis for the right region */
      x_dest = threadIdx.x + BLUR_SIZE + dim_x;
      /* The position at the image data is (x+dim_x,y) */
      idx_src = (y * width + x + dim_x) * 3 + k;

      /* Check if the position is valid and if so, performs the copy, otherwise
         just mark this position as invalid in the mask */
      if(x + dim_x < width) {
        ds_mask[x_dest][y_dest] = 1;
        ds_img[x_dest][y_dest] = inputImage[idx_src];
      } else {
        ds_mask[x_dest][y_dest] = 0;
      }
    }

    /* If the thread y axis is lower than the blur size, use the thread to
       copy the center top and center bottom regions of the sentinel */
    if(threadIdx.y < BLUR_SIZE) {
      /* The x axis in the shared memory is the same in both regions */
      x_dest = threadIdx.x + BLUR_SIZE;

      /* The y axis for the top region */
      y_dest = threadIdx.y;
      /* The position at the image data is (x,y-BLUR_SIZE) */
      idx_src = ((y - BLUR_SIZE) * width + x) * 3 + k;

      /* Check if the position is valid and if so, performs the copy, otherwise
         just mark this position as invalid in the mask */
      if(y - BLUR_SIZE > -1) {
        ds_mask[x_dest][y_dest] = 1;
        ds_img[x_dest][y_dest] = inputImage[idx_src];
      } else {
        ds_mask[x_dest][y_dest] = 0;
      }

      /* The y axis for the bottom region */
      y_dest = threadIdx.y + BLUR_SIZE + dim_y;
      /* The position at the image data is (x,y+dim_y) */
      idx_src = ((y + dim_y) * width + x) * 3 + k;

      /* Check if the position is valid and if so, performs the copy, otherwise
         just mark this position as invalid in the mask */
      if(y + dim_y < height) {
        ds_mask[x_dest][y_dest] = 1;
        ds_img[x_dest][y_dest] = inputImage[idx_src];
      } else {
        ds_mask[x_dest][y_dest] = 0;
      }
    }

    /* Use the first threads of size BLUR_SIZExBLUR_SIZE to copy the corners
       content of the image and fill the remaining space in the shared memory */
    if(threadIdx.x < BLUR_SIZE && threadIdx.y < BLUR_SIZE) {
      /* Upper-left corner indexes */
      x_dest = threadIdx.x;
      y_dest = threadIdx.y;
      idx_src = ((y - BLUR_SIZE) * width + x - BLUR_SIZE) * 3 + k;

      /* Check if the position is valid and if so, performs the copy, otherwise
         just mark this position as invalid in the mask */
      if(x - BLUR_SIZE > -1 && y - BLUR_SIZE > -1) {
        ds_mask[x_dest][y_dest] = 1;
        ds_img[x_dest][y_dest] = inputImage[idx_src];
      } else {
        ds_mask[x_dest][y_dest] = 0;
      }

      /* Upper-right corner indexes */
      x_dest = threadIdx.x + BLUR_SIZE + dim_x;
      y_dest = threadIdx.y;
      idx_src = ((y - BLUR_SIZE) * width + x + dim_x) * 3 + k;

      /* Check if the position is valid and if so, performs the copy, otherwise
         just mark this position as invalid in the mask */
      if(x + dim_x < width && y - BLUR_SIZE > -1) {
        ds_mask[x_dest][y_dest] = 1;
        ds_img[x_dest][y_dest] = inputImage[idx_src];
      } else {
        ds_mask[x_dest][y_dest] = 0;
      }

      /* Bottom-left corner indexes */
      x_dest = threadIdx.x;
      y_dest = threadIdx.y + BLUR_SIZE + dim_y;
      idx_src = ((y + dim_y) * width + x - BLUR_SIZE) * 3 + k;

      /* Check if the position is valid and if so, performs the copy, otherwise
         just mark this position as invalid in the mask */
      if(x - BLUR_SIZE > -1 && y + dim_y < height) {
        ds_mask[x_dest][y_dest] = 1;
        ds_img[x_dest][y_dest] = inputImage[idx_src];
      } else {
        ds_mask[x_dest][y_dest] = 0;
      }

      /* Bottom-right corner indexes */
      x_dest = threadIdx.x + BLUR_SIZE + dim_x;
      y_dest = threadIdx.y + BLUR_SIZE + dim_y;
      idx_src = ((y + dim_y) * width + x + dim_x) * 3 + k;

      /* Check if the position is valid and if so, performs the copy, otherwise
         just mark this position as invalid in the mask */
      if(x + dim_x < width && y + dim_y < height) {
        ds_mask[x_dest][y_dest] = 1;
        ds_img[x_dest][y_dest] = inputImage[idx_src];
      } else {
        ds_mask[x_dest][y_dest] = 0;
      }
    }

    /* Synchronize the threads as the next step can cause race conditions if
       not all the content is copied into the shared memory */
    __syncthreads();

    /* Position (x,y) must be inside the image to avoid out of the memory access,
       as thread index can't be negative, only check for width and height */
    if(x < width && y < height) {
      /* Go through each point in the image to calculate the average, this time
         using the shared memory content */
      for(i = threadIdx.x; i <= threadIdx.x + (BLUR_SIZE * 2); ++i) {
        for(j = threadIdx.y; j <= threadIdx.y + (BLUR_SIZE * 2); ++j) {
            sum += ds_img[i][j] * ds_mask[i][j];
            counter += ds_mask[i][j];
        }
      }

      /* Finally, store the result in the output image */
      outputImage[(y * width + x) * 3 + k] = (unsigned char)(sum / counter);
    }

    /* Synchronize the threads after the kernel execution for this channel */
    __syncthreads();
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  unsigned char *hostInputImageData;
  unsigned char *hostOutputImageData;
  unsigned char *deviceInputImageData;
  unsigned char *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 1);
  printf( "imagem de entrada: %s\n", inputImageFile );

//  inputImage = wbImportImage(inputImageFile);
  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

// NOW: input and output images are RGB (3 channel)
  outputImage = wbImage_new(imageWidth, imageHeight, 3);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(unsigned char) * 3);
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(unsigned char) * 3);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
 cudaMemcpy(deviceInputImageData, hostInputImageData,
            imageWidth * imageHeight * sizeof(unsigned char) * 3,
            cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");

  dim3 grid(
    (imageWidth - 1) / BLOCK_DIM_X + 1, (imageHeight - 1) / BLOCK_DIM_Y + 1, 1
  );

  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

  blurImageShm<<<grid, block>>>(
    deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight
  );

  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(unsigned char) * 3,
             cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);
  // DEBUG: if you want to see your image, 
  //   will generate file bellow in current directory
  wbExport( "blurred.ppm", outputImage );

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
