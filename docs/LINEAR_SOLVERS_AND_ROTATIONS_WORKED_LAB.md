# Numerical linear solvers and 3D rotations: work the equations, then read the C++23

This study path connects [LinearSolve.cppm](../LinearSolve.cppm), [Quaternion.cppm](../Quaternion.cppm), [Matrix.cppm](../Matrix.cppm) and [Decomposition.cppm](../Decomposition.cppm). It is designed as a sequence of hand calculations and source-reading exercises rather than a directory inventory.

## 1. First: what is a linear system?

A linear system is an equation \(Ax=b\): A is a matrix of coefficients, x is the vector of unknowns and b is the observed target. Solving it is different from explicitly calculating the inverse matrix. For most applications you need one x, not all columns of A inverse.

For example, consider

\[
A=\begin{bmatrix}2&1\\1&3\end{bmatrix},
\quad b=\begin{bmatrix}1\\2\end{bmatrix}.
\]

From \(2x+y=1\), \(y=1-2x\). Substituting into \(x+3y=2\) yields \(x=0.2,y=0.6\). A numerical solver should reproduce that answer within floating-point tolerance. The residual \(r=b-Ax\) should be near zero; however, a small residual does not by itself guarantee a small error in x for an ill-conditioned matrix.

## 2. Why a triangular system is simpler

If L is lower triangular, row i depends only on unknown x[i] and already-computed x[0],...,x[i-1]. Forward substitution computes

\[
x_i=\frac{b_i-\sum_{j<i}L_{ij}x_j}{L_{ii}}.
\]

Backward substitution in upper-triangular U works in reverse order:

\[
x_i=\frac{b_i-\sum_{j>i}U_{ij}x_j}{U_{ii}}.
\]

These operations take \(O(n^2)\) arithmetic for a dense n×n triangular system, cheaper than factoring a general dense matrix. [ForwardSubstitution](../LinearSolve.cppm) and BackwardSubstitution reject non-square or dimension-mismatched data and use a scale-relative small-pivot cutoff rather than dividing by a near-zero diagonal.

**Exercise:** construct a lower-triangular 3×3 system with solution x=[1,2,3], multiply L*x to generate b, then recover x. Make one diagonal zero and check the documented rejection path.

## 3. Pivoted Gaussian elimination

Gaussian elimination converts the augmented matrix \([A|b]\) to an upper-triangular form by subtracting multiples of the current pivot row. **Partial pivoting** selects the row with the largest absolute coefficient in the current column among rows not yet eliminated. Swapping that entire augmented row preserves the relationship between coefficients and b.

The implementation's steps are:

1. Validate square A and matching right-hand-side size.
2. Calculate a scale-aware pivot threshold.
3. Copy A and b into a single augmented matrix.
4. For each column, select maximum-absolute pivot, swap rows, reject near-zero pivot, then eliminate below.
5. Extract U and transformed b and invoke BackwardSubstitution.

For a matrix starting with a tiny leading pivot, such as

\[
A=\begin{bmatrix}10^{-12}&1\\1&1\end{bmatrix},\quad b=A\begin{bmatrix}1\\1\end{bmatrix},
\]

partial pivoting swaps the two rows before dividing. Otherwise, division by \(10^{-12}\) amplifies rounding error. Even partial pivoting is not universally stable for every possible ill-conditioned matrix; verify error and conditioning separately.

**Exercise:** print the pivot row choices, compute the residual norm \(\|Ax-b\|\) and compare with a reference double-precision solve for several scales.

## 4. One factorization for many right-hand sides

When solving \(AX=B\) with many columns, recomputing Gaussian elimination for each B column wastes work. LUP factorization represents a row-permuted A as

\[
PA=LU.
\]

For each b column: permute it to Pb, solve Ly=Pb by forward substitution, then solve Ux=y by backward substitution. KairoMath's matrix-RHS LinearSolve computes LUP once and repeats only the triangular solves.

The Inverse function similarly solves \(Ax=e_j\) for every identity column, assembling A inverse. Avoid forming that full inverse when only one \(Ax=b\) is required. Compare runtime, roundoff and residuals for n=8, n=64 and n=256 before drawing speed conclusions.

## 5. Orientation is not position: understand quaternions

Kairo's quaternion holds \((x,y,z,w)\), with xyz the vector part and w the scalar. For a unit axis a and angle \(\theta\) radians, AxisAngle creates

\[
q=(a_x\sin(\theta/2),a_y\sin(\theta/2),
   a_z\sin(\theta/2),\cos(\theta/2)).
\]

A unit quaternion describes a pure rotation. Nonunit quaternions need normalization, and the algebraic inverse is \(q^{-1}=\operatorname{conj}(q)/\|q\|^2\). The implementation returns identity when normalization/inversion encounters very small magnitude. That is an explicit fallback, not proof that the original input was a valid orientation.

For a vector v embedded as pure quaternion \((v_x,v_y,v_z,0)\), Rotate computes

\[
v' = q v q^{-1}.
\]

The fast path for nearly unit q replaces inverse with conjugate. Quaternion::ToMatrix3 normalizes q before producing the rotation matrix.

**Exercise:** use an axis-angle quaternion for 90° about +Y, rotate a forward-vector and check orientation against ToMatrix3. Verify lengths remain equal and compare q with -q; both encode the same rotation.

## 6. Matrix layout and operation order

[Matrix.cppm](../Matrix.cppm) uses **row-major storage** for elements while mathematics uses **column vectors**. Do not confuse memory layout with multiplication convention.

A product \(Av\) applies A to v; \(ABv\) applies B first, then A. Translation lives in the last column of a 4×4 affine matrix. Shaders and GPU upload adapters must honor the selected memory packing convention; using the wrong transpose can produce a scene that looks plausible but is geometrically wrong.

**Exercise:** write a translation and a rotation; compute Rotation*Translation*point and Translation*Rotation*point and explain why they differ. Keep point w=1 and direction w=0 distinct.

## 7. Camera-facing and animation interpolation

LookRotation constructs an orthonormal basis from requested forward and approximate up. In this engine, local **−Z is forward**, so the rotation matrix columns represent right, up and negative forward. If forward is zero, the source returns identity; if up and forward are collinear, it chooses a fallback orthogonal direction.

Quaternions q and −q represent one orientation. NLerp flips the second endpoint if the dot product is negative, linearly blends and normalizes. SLerp uses spherical sine weights to follow a shorter great-circle path, falling back to NLerp when endpoints are nearly identical to avoid a tiny denominator.

**Exercise:** interpolate between identity and a half-turn about Y with alpha=0, 0.25, 0.5, 0.75, 1. Check output quaternion norm, angle progression and equivalence when replacing the endpoint q with -q. SLerp generally better approximates uniform angular speed; NLerp is cheaper.

## 8. Accuracy boundaries and downstream projects

LinearSolve/Decomposition feed optimization, physics constraints and numerical reconstruction. Quaternion/Matrix/Transform feed engine entity transforms, KairoRenderer cameras, physics angular integration and glTF animation.

The key distinction is that a numerical result can have **small residual**, **small parameter error**, **stable geometry**, or **good runtime**; none implies all others automatically. Record condition estimates where applicable, floating-point type, input scale, epsilon policy, exact test source and compiler.

Read the repository's [full source guide](SOURCE_READING_ORDER.md) and its matrix, decomposition and quaternion tests. Add comparisons for singular inputs, large/small pivots, many RHS columns, Euler order, nonunit quaternions, ±q sign, and near-collinear LookRotation axes.

*This is source-based teaching documentation. No new C++ build or numerical benchmark was run as part of this annotation pass.*
