#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include "implicit/gpu/als.h"
#include "implicit/gpu/utils.cuh"
#include <ostream>
#include <fstream>      // std::filebuf

namespace implicit {

__global__ void lmf_update_kernel(float* _deriv, int num_items,
                                  int begin_index, int end_index,
                                  float* deriv_sum,
                                  unsigned int * random_dislikes,
                                  int * indices, int * indptr, float * data,
                                  int factors,
                                  float * X, float * Y,
                                  float learning_rate, float reg,
                                  long neg_prop) {

    extern __shared__ float shared_memory[];
    float * temp = &shared_memory[0];
    int neg_processed = 0;

    for(int userid = begin_index + blockIdx.x; userid < end_index; userid += gridDim.x) {

        int user_seen_item = indptr[userid + 1] - indptr[userid];
        if (user_seen_item == 0)
            continue;
        float * deriv = &_deriv[userid * factors];
        memset(deriv, 0, sizeof(float) * factors);
        float * vec_deriv_sum = &deriv_sum[userid * factors];

        // [indptr[i], indptr[i +1])까지가 user가 consume한 positive items;

        int likedid, dislikedid;

        float score;
        float * user = &X[userid * factors];
        float z;
        float user_val = user[threadIdx.x];

        for (int i=indptr[userid]; i < indptr[userid + 1]; ++i){
            likedid = indices[i];
            float * liked = &Y[likedid * factors];
            score = exp(dot(user, liked));
            float z = data[i] * (score/ (1.0 + score));
            float liked_val = liked[threadIdx.x];
            deriv[threadIdx.x] += (data[i] - z) * liked_val;
        }

        for (int i = neg_prop * indptr[userid]; i < neg_prop * indptr[userid + 1]; ++i) {
            dislikedid = random_dislikes[i] % num_items;
            float * disliked = &Y[dislikedid * factors];
            float score = 0;
            score += user[threadIdx.x] * disliked[threadIdx.x];
            score = exp(score);

            float z = (score / (1.0 + score));
            deriv[threadIdx.x] -= z * disliked[threadIdx.x];

        }
        /*
        deriv[threadIdx.x] -= reg * user_val;
        vec_deriv_sum[threadIdx.x] += deriv[threadIdx.x] * deriv[threadIdx.x];
        user[threadIdx.x] += (learning_rate / (sqrt(1e-6 + vec_deriv_sum[threadIdx.x]))) * deriv[threadIdx.x];
        */
    }

}

std::pair<int, int>  lmf_update(int * c_indptr,
                                CudaDenseMatrix * deriv_sum,
                                CudaDenseMatrix *X,
                                CudaDenseMatrix *Y,
                                const CudaVector<int>& indices,
                                const CudaVector<int>& indptr,
                                const CudaVector<float>& data,
                                float learning_rate,
                                float reg,
                                long neg_prop,
                                long seed) {
    if (X->cols != Y->cols) throw std::invalid_argument("X and Y should have the same number of columns");
    /*
    if (userids.size != itemids.size)
        throw std::invalid_argument("userids and itemids should have same number of elements");
    */
   std::filebuf fb;
   fb.open("tmp.txt", std::ios::out);
   std::ostream os(&fb);

   if (indptr.size != X->rows + 1)
        throw std::invalid_argument("indptr has some error");

    // todo: check indptr = X->rows + 1
    const int num_users = X->rows;
    const int num_items = Y->rows;
    os<<num_users<<' '<<num_items<<'\n';

    // allocate some memory
    std::default_random_engine rng_engine (seed);
    std::uniform_int_distribution<unsigned int> item_rng(0, num_items - 1);
    std::vector<unsigned int> rng_vector;
    float* _deriv;
    CHECK_CUDA(cudaMalloc(&_deriv, sizeof(float) * num_users * X->cols));
    //CHECK_CUDA(cudaMalloc(&stats, sizeof(int) * 2));





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

    const int num_samples = (1 << 10);
    // initialize memory for randomly picked positive/negative items
    unsigned int * random_dislikes;
    CHECK_CUDA(cudaMalloc(&random_dislikes, 2 * num_samples * neg_prop * sizeof(unsigned int)));
    // TODO: get rows passed in here
    //CHECK_CURAND(curandGenerate(rng, random_dislikes, 5 * neg_prop));
    //CHECK_CURAND(curandGenerate(rng, random_dislikes, 10 * neg_prop));
    //CHECK_CURAND(curandGenerate(rng, random_dislikes, 20 * neg_prop));
    rng_vector.resize(num_samples * neg_prop);

    int begin_index = 0;
    int end_index = 0;

    while (end_index < num_users) {
        int num_positives = c_indptr[end_index] - c_indptr[begin_index];
        os<<begin_index<<' '<<end_index<<' '<<num_positives<<"\n";
        if (num_positives >= num_samples) {
            os<<"hit"<<"\n";
            CHECK_CURAND(curandGenerate(rng, random_dislikes, num_samples * neg_prop));

            lmf_update_kernel<<<block_count, thread_count, shared_memory_size>>>(
                _deriv, num_items,
                begin_index, end_index,
                deriv_sum->data,
                random_dislikes,
                indices.data, indptr.data, data.data,
                factors,
                X->data, Y->data,
                learning_rate, reg,
                neg_prop);
            begin_index = end_index;
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        end_index += 1;
    }

    if (begin_index < num_users) {
        os<<"?"<<"\n";
        os<<begin_index<<' '<<num_users<<'\n';
        fb.close();
        CHECK_CURAND(curandGenerate(rng, random_dislikes, num_samples * neg_prop));
        lmf_update_kernel<<<block_count, thread_count, shared_memory_size>>>(
            _deriv, num_items,
            begin_index, num_users,
            deriv_sum->data,
            random_dislikes,
            indices.data, indptr.data, data.data,
            factors,
            X->data, Y->data, learning_rate, reg,
            neg_prop);

    }



    // we're returning the number of correctly ranked items, get that value from the device
    /*
    int output[2];
    CHECK_CUDA(cudaMemcpy(output, stats, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(random_dislikes));
    CHECK_CUDA(cudaFree(stats));
    curandDestroyGenerator(rng);
    */
   //int output[2] = {0, 0};
    return {0,0};

}
}
