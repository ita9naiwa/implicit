#ifndef IMPLICIT_GPU_LMF_H_
#define IMPLICIT_GPU_LMF_H_
#include "implicit/gpu/matrix.h"
#include <utility>

namespace implicit {
std::pair<int, int>  lmf_update(CudaDenseMatrix * vec_deriv_sum,
                                CudaDenseMatrix *X,
                                CudaDenseMatrix *Y,
                                const CudaVector<int>& indices,
                                const CudaVector<int>& indptr,
                                const CudaVector<float>& data,
                                float learning_rate,
                                float regularization,
                                long neg_prop,
                                long seed);
}  // namespace implicit
#endif  // IMPLICIT_GPU_LMF_H_
