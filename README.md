# PCA-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD
Mini Project - Face Detection or Convert an image into gray scale image using CUDA GPU programming
### AIM : 
To write a CUDA C program to convert color image into grey scale image in GPU.
### PROCEDURE:

#### STEP 1:
Include the necessary header files, such as stdio.h, stdlib.h, and windows.h for Windows-specific timing.
#### STEP 2:
Define constants for the image dimensions, number of channels, and block size.
#### STEP 3:
Declare the kernel function colorToBW that converts the color image to black and white.
#### STEP 4:
In the `main` function:
   - Specify the input and output image file names.
   - Read the input image file into memory.
   - Allocate memory for the output image.
   - Allocate device memory for input and output images using cudaMalloc.
   - Copy the input image data from host to device using cudaMemcpy.
   - Set the grid and block dimensions for the kernel execution.
   - Start the timer using QueryPerformanceFrequency and QueryPerformanceCounter.
   - Launch the kernel function using <<<dimGrid, dimBlock>>> syntax.
   - Copy the output image data from device to host using cudaMemcpy.
   - Stop the timer and calculate the elapsed time.
   - Write the output image to a file.
   - Free the device and host memory.
   - Print the completion message and the elapsed time.
### PROGRAM : 
```
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define WIDTH 512
#define HEIGHT 512
#define CHANNELS 3
#define BLOCK_SIZE 16

__global__ void colorToBW(unsigned char* input, unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int tid = y * width + x;
        unsigned char r = input[tid * CHANNELS];
        unsigned char g = input[tid * CHANNELS + 1];
        unsigned char b = input[tid * CHANNELS + 2];

        unsigned char bw = 0.21f * r + 0.71f * g + 0.07f * b;

        output[tid] = bw;
    }
}

int main()
{
    const char* inputImageFile = "mp.jpg";
    const char* outputImageFile = "output.jpg";

    // Read the input image file
    FILE* file = fopen(inputImageFile, "rb");
    if (!file) {
        printf("Error: Could not open input image file.\n");
        return 1;
    }

    fseek(file, 0, SEEK_END);
    int imageSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    unsigned char* inputImage = (unsigned char*)malloc(sizeof(unsigned char) * imageSize);
    fread(inputImage, sizeof(unsigned char), imageSize, file);
    fclose(file);

    // Determine image dimensions
    int width = WIDTH;
    int height = HEIGHT;

    // Allocate memory for output image
    int outputSize = width * height;
    unsigned char* outputImage = (unsigned char*)malloc(sizeof(unsigned char) * outputSize);

    // Allocate device memory
    unsigned char* d_inputImage, * d_outputImage;
    cudaMalloc((void**)&d_inputImage, sizeof(unsigned char) * imageSize);
    cudaMalloc((void**)&d_outputImage, sizeof(unsigned char) * outputSize);

    // Copy input image data from host to device
    cudaMemcpy(d_inputImage, inputImage, sizeof(unsigned char) * imageSize, cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    // Start timer
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    // Launch kernel
    colorToBW << <dimGrid, dimBlock >> > (d_inputImage, d_outputImage, width, height);

    // Copy output image data from device to host
    cudaMemcpy(outputImage, d_outputImage, sizeof(unsigned char) * outputSize, cudaMemcpyDeviceToHost);

    // Stop timer
    QueryPerformanceCounter(&end);
    double elapsed_time = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    // Write the output image file
    FILE* outputFile = fopen(outputImageFile, "wb");
    if (!outputFile) {
        printf("Error: Could not open output image file.\n");
        return 1;
    }

    fwrite(outputImage, sizeof(unsigned char), outputSize, outputFile);

    fclose(outputFile);

    // Free device memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Free host memory
    free(inputImage);
    free(outputImage);

    printf("Image converted to black and white successfully!\n");
    printf("Output image file: %s\n", outputImageFile);
    printf("Elapsed Time: %.6f seconds\n", elapsed_time);

    return 0;
}

```
### OUTPUT : 
#### input.jpg 
![input](https://github.com/SOWMIYA2003/PCA---Mini-Project-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD/assets/93427443/674ffc6f-6891-4470-ba44-74da67dbacdb)
#### output.jpg
![output](https://github.com/SOWMIYA2003/PCA---Mini-Project-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD/assets/93427443/da81a6db-52e9-4459-84bb-22575ae831e3)
### RESULT :
Thus a CUDA C Program is executed successfully to convert color image into grey scale image in GPU.
