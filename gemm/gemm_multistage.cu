/**
 * 多阶流水线实现的通用矩阵乘法，实现参考CUTLASS模板库。
 */

namespace gemm_multistage {

template <int M>
struct GemmShape {

};

template <
    typename ElementA,
    typename LayoutA,
    typename ElementB,
    typename LayoutB,
    typename ElementC,
    typename LayoutC,
    typename ElementAccumulator = ElementC
>
class GemmMultistage {


};

} // namespace gemm_multistage