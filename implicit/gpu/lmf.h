#ifndef IMPLICIT_GPU_LMF_H_
#define IMPLICIT_GPU_LMF_H_
#include "implicit/gpu/matrix.h"
#include <utility>

namespace implicit {
std::pair<int, int>  lmf_update(CudaDenseMatrix * vec_deriv_sum,
                                const CudaVector[int]& indices,
                                const CudaVector[int]& indptr,
                                const CudaVector[int]& data,
                                CudaDenseMatrix *X,
                                CudaDenseMatrix *Y,
                                float learning_rate,
                                float regularization,
                                long neg_prop,
                                long seed);
}  // namespace implicit
#endif  // IMPLICIT_GPU_LMF_H_
