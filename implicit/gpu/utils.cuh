#ifndef IMPLICIT_GPU_UTILS_CUH_
#define IMPLICIT_GPU_UTILS_CUH_
#include <stdexcept>
#include <sstream>

namespace implicit {
using std::invalid_argument;

// Error Checking utilities, checks status codes from cuda calls
// and throws exceptions on failure (which cython can proxy back to python)

#define CHECK_CUDA(code) { checkCuda((code), __FILE__, __LINE__); }
inline void checkCuda(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::stringstream err;
        err << "Cuda Error: " << cudaGetErrorString(code) << " (" << file << ":" << line << ")";
        throw std::runtime_error(err.str());
    }
}

inline const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "Unknown";
}

#define CHECK_CUBLAS(code) { checkCublas((code), __FILE__, __LINE__); }
inline void checkCublas(cublasStatus_t code, const char * file, int line) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        std::stringstream err;
        err << "cublas error: " << cublasGetErrorString(code)
            << " (" << file << ":" << line << ")";
        throw std::runtime_error(err.str());
    }
}


#define WARP_SIZE 32

// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
__inline__ __device__
float warp_reduce_sum(float val) {

#if __CUDACC_VER_MAJOR__ >= 9
    // __shfl_down is deprecated with cuda 9+. use newer variants
    unsigned int active = __activemask();
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(active, val, offset);
    }
#else
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
#endif
    return val;
}

__inline__ __device__
float dot(const float * a, const float * b) {
    __syncthreads();
    static __shared__ float shared[32];

    // figure out the warp/ position inside the warp
    int warp =  threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // partially reduce the dot product inside each warp using a shuffle
    float val = a[threadIdx.x] * b[threadIdx.x];
    val = warp_reduce_sum(val);

    // write out the partial reduction to shared memory if appropiate
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    // if we we don't have multiple warps, we're done
    if (blockDim.x <= WARP_SIZE) {
        return shared[0];
    }

    // otherwise reduce again in the first warp
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (warp == 0) {
        val = warp_reduce_sum(val);
        // broadcast back to shared memory
        if (threadIdx.x == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    return shared[0];
}

// TODO: we could use an n-ary search here instead, but
// that will only be faster when the number of likes for a user is
// much greater than the number of threads (factors) we are using.
// Since most users on most datasets have relatively few likes, I'm
// using a simple linear scan here instead.
__inline__ __device__
bool linear_search(int * start, int * end, int target) {
    __shared__ bool ret;

    if (threadIdx.x == 0) ret = false;
    __syncthreads();

    int size = end - start;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (start[i] == target) {
            ret = true;
        }
    }
    __syncthreads();
    return ret;
}

#define CHECK_CURAND(code) { checkCurand((code), __FILE__, __LINE__); }
inline void checkCurand(curandStatus_t code, const char *file, int line) {
    if (code != CURAND_STATUS_SUCCESS) {
        std::stringstream err;
        err << "CURAND error: " << code << " (" << file << ":" << line << ")";
        throw std::runtime_error(err.str());
    }
}


}  // namespace implicit
#endif  // IMPLICIT_GPU_UTILS_CUH_
