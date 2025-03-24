#ifndef PHASMA_MATRIX_FREE_OPERATOR_HPP
#define PHASMA_MATRIX_FREE_OPERATOR_HPP

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace Phasma::EigenSupport {
// Implementation of the matrix-free wrapper class.
template <typename Scalar_>
class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement<Scalar_>> {
public:
  // Required typedefs, constants, and method:
  using Scalar = Scalar_;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using StorageIndex = int;
  using MatrixOp = std::function<void(const Eigen::Ref<const Phasma::Vector<Scalar>>&, 
                                      Eigen::Ref<Phasma::Vector<Scalar>>)>;

  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };

  Eigen::Index rows() const { return rows_; }
  Eigen::Index cols() const { return cols_; }

  template <typename Rhs>
  Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

  // Custom API:
  MatrixReplacement(const MatrixOp& op, Eigen::Index rows, Eigen::Index cols) 
      : op_(op), rows_(rows), cols_(cols) {}

  void matvec(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& result) const {
    op_(x, result);
  }

private:
  const MatrixOp& op_;
  const Eigen::Index rows_, cols_;
};

} // namespace Phasma

namespace Eigen {
namespace internal {

// MatrixReplacement looks like a SparseMatrix, so let's inherit its traits:
template <typename Scalar>
struct traits<Phasma::EigenSupport::MatrixReplacement<Scalar>> 
    : public Eigen::internal::traits<Eigen::SparseMatrix<Scalar>> {};

template <typename Scalar, typename Rhs>
struct generic_product_impl<Phasma::EigenSupport::MatrixReplacement<Scalar>, Rhs, SparseShape, DenseShape, GemvProduct> 
    : generic_product_impl_base<Phasma::EigenSupport::MatrixReplacement<Scalar>, Rhs, generic_product_impl<Phasma::EigenSupport::MatrixReplacement<Scalar>, Rhs>> {
  using ScalarType = typename Product<Phasma::EigenSupport::MatrixReplacement<Scalar>, Rhs>::Scalar;

  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const Phasma::EigenSupport::MatrixReplacement<Scalar>& lhs, const Rhs& rhs, const ScalarType& alpha) {
    // This method should implement "dst += alpha * lhs * rhs" in-place,
    // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
    assert(alpha == ScalarType(1) && "Scaling is not implemented");
    EIGEN_ONLY_USED_FOR_DEBUG(alpha);

    lhs.matvec(rhs, dst);
  }
};

} // namespace internal
} // namespace Eigen

#endif // PHASMA_MATRIX_FREE_OPERATOR_HPP
