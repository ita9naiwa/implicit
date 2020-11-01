#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>


#include "implicit/gpu/als.h"
#include "implicit/gpu/utils.cuh"

namespace implicit {

__global__ void lmf_update_kernel(int begin_index, int end_index,
                                  float* vec_deriv_sum,
                                  int samples,
                                  unsigned int * random_dislikes,
                                  int * indices, int * indptr, int * data,
                                  int factors,
                                  float * X, float * Y,
                                  float learning_rate, float reg,
                                  long neg_prop,
                                  int * stats) {
    extern __shared__ float shared_memory[];
    float * temp = &shared_memory[0];
    float deriv[factors];
    memset(deriv, 0, sizeof(float) * factors);
    int neg_processed = 0;

    for(int userid = begin_index + blockIdx.x; i < end_index; userid += gridDim.x) {
        int user_seen_item = indptr[userid + 1] - indptr[userid];
        if (user_seen_item == 0)
            continue;

        float * deriv_sum_sq = &vec_deriv_sum[userid * factors];
        // [indptr[i], indptr[i +1])까지가 user가 consume한 positive items;

        int likedid, dislikedid;

        float score;
        float * user = &X[userid * factors];
        for (int i=indptr[userid]; i < indptr[userid + 1]; ++i){
            likedid = indices[i];
            float * liked = &Y[likedid * factors];
            score = exp(dot(user, liked));
            float z = data[i] * (score/ (1.0 + score));
            float user_val = user[threadIdx.x],
                  liked_val = liked[threadIdx.x];
            deriv[threadIdx.x] += (data[i] - z) * liked[threadIdx.x]
        }

        for (int i = neg_prop * indptr[i]; neg_prop * indptr[j + 1]; ++j) {
            dislikedid = random_dislikes[i];
            float * dislikedid = &Y[dislikedid * factors];
            score = exp(dot(user, disliked));
            z = (score / (1.0 + score));
            deriv[threadIdx.x] -= z * disliked[threadIdx.x];
        }
        deriv[threadIdx.x] -= regularization * user[threadIdx.x];
        deriv_sum_sq[threadIdx.x] += deriv[threadIdx.x] * deriv[threadIdx.x];
        user[threadIdx.x] += (learning_rate * / (sqrt(1e-6 + deriv_sum_sq[threadIdx.x]))) * deriv[threadIdx.x];
    }
}

std::pair<int, int>  lmf_update(CudaDenseMatrix * vec_deriv_sum,
                                const CudaVector[int]& indices,
                                const CudaVector[int]& indptr,
                                const CudaVector[int]& data,
                                CudaDenseMatrix *X,
                                CudaDenseMatrix *Y,
                                float learning_rate,
                                float regularization,
                                long neg_prop,
                                long seed) {
    if (X->cols != Y->cols) throw std::invalid_argument("X and Y should have the same number of columns");
    if (userids.size != itemids.size)
        throw std::invalid_argument("userids and itemids should have same number of elements");
    // todo: check indptr = X->rows + 1

    int num_users = X.rows
    int nonzeros = userids.size;

    // allocate some memory
    int * stats;
    CHECK_CUDA(cudaMalloc(&stats, sizeof(int) * 2));
    CHECK_CUDA(cudaMemset(stats, 0, sizeof(int) * 2));

    // initialize memory for randomly picked positive/negative items
    unsigned int * random_dislikes;


    // Create a seeded RNG
    curandGenerator_t rng;
    CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));

    // Randomly pick values


    // TODO: multi-gpu support
    int devId;
    CHECK_CUDA(cudaGetDevice(&devId));

    int multiprocessor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count,
                                      cudaDevAttrMultiProcessorCount,
                                      devId));

    int factors = X->cols;
    int block_count = 128 * multiprocessor_count;
    int thread_count = factors;
    int shared_memory_size = sizeof(float) * (factors);
    int begin_index = 0, end_index = 0;
    int num_samples = (1 << 16);
    CHECK_CUDA(cudaMalloc(&random_dislikes, num_samples * neg_prop * sizeof(unsigned int)));
    // TODO: get rows passed in here
    while (end_index < num_users) {
        int num_positives = indptr[end_index] - indptr[begin_index];
        if (num_positives >= num_samples)
        {
            CHECK_CURAND(curandGenerate(rng, random_dislikes, num_samples * neg_prop));
            lmf_update_kernel<<<block_count, thread_count, shared_memory_size>>>(
                begin_index, end_index,
                deriv_sum_sq.data,
                nonzeros,
                random_dislikes,
                indices.data, indptr.data, data.data,
                factors,
                X->data, Y->data, learning_rate, reg,
                neg_prop,
                stats);
            begin_index = end_index;
            CHECK_CUDA(cudaDeviceSynchronize());
        } else {
            end_index = += 1;
        }
    }
    if (begin_index != end_index) {
        CHECK_CURAND(curandGenerate(rng, random_dislikes, num_samples * neg_prop));
        lmf_update_kernel<<<block_count, thread_count, shared_memory_size>>>(
            begin_index, end_index,
            deriv_sum_sq.data,
            nonzeros,
            random_dislikes,
            indices.data, indptr.data, data.data,
            factors,
            X->data, Y->data, learning_rate, reg,
            neg_prop,
            stats);
    }

    // we're returning the number of correctly ranked items, get that value from the device
    int output[2];
    CHECK_CUDA(cudaMemcpy(output, stats, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(random_dislikes));
    CHECK_CUDA(cudaFree(stats));
    curandDestroyGenerator(rng);
    return std::make_pair(output[0], output[1]);
}
}
