// matrix transpose, 3 versions:
// 1) baseline
// 2) optimized
// 3) LDS


__global__ void transpose_baseline(const float* in,
                                    float* out,
                                    int width,
                                    int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        out[col*height + row] = in[row * width + col];

        // 1. consecutive col values access consecutive input elements
        // 2. consecutive col values access output elements height apart
    }
}

__shared__ float tile[TILE][TILE];
    int in_col = blockIdx.x * TILE + threadIdx.x;
    int in_row = blockIdx.y * TILE + threadIdx.y;
    if (in_col < width && in_row < height)
        tile[threadIdx.y][threadIdx.x] = 
            in[in_row * width + in_col]; // coalesced ReadableBuffer
    __syncthreads();


// optimized kernel

#define TILE 16
__global__ void transpose_optimized(const float* in, float* out, int width, int height) {
    __shared__ float tile[TILE][TILE+1]; // +1 padding reduces LDS bank conflicts

    // input coordinates in the original height x width matrix
    int in_col = blockIdx.x * TILE + threadIdx.x;
    int in_row = blockIdx.y * TILE + threadIdx.y;

    // load the input tile row-wise into LDS
    if (in_col < width && in_row < height) {
        tile[threadIdx.y][threadIdx.x] = in[in_row * width + in_col];        
    }

    __syncthreads(); // the entire workgroup must finish loading the tile

    // swap the block coordinates for the transposed loading the tile
    int out_col = blockIdx.y * TILE + threadIdx.x;
    int out_row = blockIdx.x * TILE + threadIdx.y;

    // output has width rows x height columns
    if (out_col < height && out_row < width) {
        out[out_row * height + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}