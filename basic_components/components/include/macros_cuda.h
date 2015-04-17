#ifndef CUDA_MACRO_H
#define CUDA_MACRO_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */

//#include <helper_cuda.h>

#define STRIDE_ALIGNMENT 32
// Align up n to the nearest multiple of m
#define ALIGN_UP(n) (((n) % STRIDE_ALIGNMENT) ? ((n) + STRIDE_ALIGNMENT - ((n) % STRIDE_ALIGNMENT)) : (n))

#ifdef CUDA_CODE

#define GLOBAL

#define KERNEL __global__

#define DEVICE_HOST __device__ __host__

//! HW 29/03/15 : modif
#define DEVICE __device__
#define HOST __host__

#define KER_SCHED(w,h) \
    int _x = blockIdx.x * blockDim.x + threadIdx.x;\
    int _y = blockIdx.y * blockDim.y + threadIdx.y;
#define END_KER_SCHED

#define KER_CALL_THREAD_BLOCK(b, t, b_width, b_height, width, height) \
    dim3    t(b_width, b_height);\
    dim3    b((width + t.x - 1) / t.x,(height + t.y - 1) / t.y);

#define _KER_CALL_(b,t) <<< b, t >>>

#define GPU_ALLOC_MEM(devPtr,size) (checkCudaErrors(cudaMalloc((void**)&(devPtr),(size))))

#define GPU_FREE_MEM(devPtr) (checkCudaErrors(cudaFree((devPtr))))

#define SYNCTHREADS __syncthreads();

#else//CUDA_CODE

#define GLOBAL

#define KERNEL

#define DEVICE_HOST

//! HW 29/03/15 : modif
#define DEVICE
#define HOST

#define KER_SCHED(w,h) \
    for (int _y = 0; _y < h; ++_y) {\
    for (int _x = 0; _x < w; ++_x) {\

#define END_KER_SCHED     }}

#define KER_CALL_THREAD_BLOCK(b, t, b_width, b_height, width, height) \
    int b;\
    int t;\

#define _KER_CALL_(b,t)

#define GPU_ALLOC_MEM(devPtr,size)

#define GPU_FREE_MEM(devPtr)

#define SYNCTHREADS

#endif//CUDA_CODE

#endif // CUDA_MACRO_H
