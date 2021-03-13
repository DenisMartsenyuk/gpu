__kernel void simple_multiplication( __global float* matrix_a, __global float* matrix_b, __global float* result, const unsigned int height, const unsigned int width)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    float sum = 0.0f;
    
    for (int i = 0; i < height; i++) {
        
        sum += matrix_a[row * height + i] * matrix_b[i * width + col];
    }
    result[row * width + col] = sum;
}

#define BLOCK_SIZE 16
__kernel void optimization_1_multiplication(__global float* matrix_a, __global float* matrix_b, __global float* result, const int M, const int N, const int K)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int x_in_block = get_local_id(0);
    int y_in_block = get_local_id(1);
    
    __local float sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __local float sub_b[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0;
    for (int i = 0; i < K / BLOCK_SIZE; ++i) {
        sub_a[y_in_block][x_in_block] = matrix_a[y * K + i * BLOCK_SIZE + x_in_block];
        sub_b[y_in_block][x_in_block] = matrix_b[(i * BLOCK_SIZE + y_in_block) * K + x];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            sum += sub_a[y_in_block][j] * sub_b[j][x_in_block];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    result[K * y + x] = sum;
}

