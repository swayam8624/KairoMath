# Foundation Math Roadmap

This document captures follow-up work for the Foundation Math layer so useful review notes do not get lost in chat history.

Current module chain:

```text
Vector
  -> Matrix
  -> Quaternion
  -> Transform
```

The current math layer is strong enough to support the next engine systems. Do not keep polishing math indefinitely. After the correctness items below, move into geometry and renderer stress tests.

## Completed Fixes

Completed in the current V1 stabilization pass:

- `Refract()` now guards total internal reflection and returns zero instead of producing NaNs.
- `Project()` now works for non-unit axes and returns zero for zero-length axes.
- Floating-point vector, matrix, and quaternion scalar division now uses reciprocal multiplication while preserving integer division semantics.
- Quaternion `Normalize()` is no longer `constexpr` because it depends on runtime math functions through `LengthInverse()`.
- Quaternion `FromMatrix3()` now guards near-zero divisors before converting matrix branches.
- Quaternion Euler conversion now documents its radians / pitch-yaw-roll convention.
- Quaternion `ToAxisAngle()` clamps `w` before `acos()` / denominator calculation.
- Matrix convention is documented at the top of `Matrix.cppm`.
- Transform inverse now asserts uniform scale because a general rotated non-uniform-scale inverse cannot always be represented exactly as another TRS transform.
- `WorldToLocal()` remains the correct point inverse path for non-uniform-scale transforms.
- Cross-module smoke tests verify projection, refraction, matrix inverse, uniform transform inverse, non-uniform `WorldToLocal()`, quaternion sign equivalence, and matrix/quaternion round-trip.

## Fix Now

These are correctness or confidence items worth addressing before building larger systems.

### Vector

- Done: add or verify total-internal-reflection handling in `Refract()`.
  - If refraction cannot occur, return `VectorN<T>::Zero()` or intentionally switch to `Reflect()`.
  - Avoid `sqrt(negative)` producing NaNs.

- Done: fix `Project()` for non-unit projection axes.
  - Current formula is only valid when `onto` is normalized.
  - Correct formula:

```cpp
return onto * (Dot(vector, onto) / Dot(onto, onto));
```

- Done in smoke test; still add permanent Catch2 tests for `Refract()` edge cases and `Project()` with non-unit axes.

### Matrix

- Done in smoke test; still add dedicated Catch2 tests for `Inverse(Matrix4)`.
  - Verify translation-only, rotation-only, scale-only, and combined TRS matrices.
  - Verify `matrix * Inverse(matrix)` is nearly identity.
  - Verify camera/view/picking-style matrices.

- Done: add epsilon guards in `FromMatrix3()` quaternion conversion paths where division by `s` happens.

- Done: ensure matrix convention is documented prominently:
  - row-major storage
  - column-vector multiplication
  - transform composition order
  - Vulkan-style depth range `[0, 1]` for projections

### Quaternion

- Done: document Euler convention in `ToEuler()` / `FromEuler()`.
  - radians
  - pitch = x
  - yaw = y
  - roll = z
  - exact rotation order/convention

- Done: verify no `constexpr` remains on quaternion functions that call `std::sqrt`, `std::sin`, `std::cos`, `std::acos`, etc.

### Transform

- Done in smoke test for uniform-scale `Inverse(Transform)` and non-uniform `WorldToLocal()`; still add permanent Catch2 tests.
  - Verify `Inverse(Inverse(t)) ~= t`.
  - Verify `t * Inverse(t) ~= Identity`.
  - Verify non-uniform scale cases through `WorldToLocal()` and matrix inverse, not `Inverse(Transform)`, because non-uniform rotated inverse is generally not representable as pure TRS.
  - Verify `WorldToLocal(t, TransformPoint(t, p)) ~= p`.

## Test Suite To Add

Use Catch2-style tests when the test target is set up.

### Vector Tests

- `Zero`, `One`, `UnitX/Y/Z/W`, and semantic directions.
- `Data()` pointer order.
- `Length`, `LengthSquared`, `LengthInverse`.
- `Normalize()` and `SafeNormalize()` zero-vector behavior.
- `Dot`, `Cross`, `Reflect`, `Refract`, `Project`.
- `Distance`, `DistanceSquared`, `AngleBetween`.
- `NearlyEqual()` with tolerance.

### Matrix Tests

- `Identity`, `Zero`, `One`.
- `operator[]`, `operator()`, `Row`, `Column`, `SetRow`, `SetColumn`.
- `Transpose`, `Trace`, `Determinant`.
- `Inverse(Matrix3)` and `Inverse(Matrix4)`.
- `MakeTranslation`, `MakeScale`, `MakeRotationX/Y/Z`.
- `Perspective`, `Orthographic`, `LookAt`.
- `TransformPoint`, `TransformDirection`, `TransformNormal`.
- `ExtractTranslation`, `ExtractScale`, `ExtractRotationMatrix`.

### Quaternion Tests

- `Identity`, `Zero`, `Length`, `LengthInverse`, `IsNormalized`.
- `Conjugate`, `Inverse`, `Normalize`, `Normalized`.
- `AxisAngle` with normalized and non-normalized axes.
- `RotationAroundX/Y/Z`.
- Hamilton product composition order.
- `Rotate()` with unit and non-unit quaternions.
- `ToMatrix3`, `ToMatrix4`, `FromMatrix3`, `FromMatrix4`.
- `ToEuler`, `FromEuler`.
- `Lerp`, `NLerp`, `SLerp`.
- `FromToRotation`, `LookRotation`.
- `Forward`, `Right`, `Up`, `Backward`, `Left`, `Down`.
- `NearlyEqual(q, -q)` rotation equivalence.

### Transform Tests

- Constructors and static constructors.
- Setters.
- `Forward`, `Right`, `Up`, reverse directions.
- `WorldMatrix`, `RotationMatrix`.
- `TransformPoint`, `TransformDirection`, `TransformVector`, `TransformNormal`.
- `LocalToWorld`, `WorldToLocal`.
- Parent-child composition.
- `Inverse`.
- `Interpolate`.
- `NearlyEqual`, `IsIdentity`, `IsValid`, `HasUniformScale`.

## Build Next

Build geometry primitives before camera/frustum/physics.

```text
Ray.cppm
Plane.cppm
AABB.cppm
```

These unlock:

```text
CPU ray tracing
BVH construction
Picking
Collision queries
Spatial queries
Frustum culling groundwork
```

Recommended dependency direction:

```text
Vector
  -> Matrix
  -> Quaternion
  -> Transform
  -> Ray / Plane / AABB
  -> Frustum
  -> Camera
  -> BVH
  -> CPU Ray Tracer
```

## Later Foundation Features

Do these after the geometry modules begin stress-testing the current math layer.

- `DecomposeTRS()`.
- `InverseTranspose3x3()` or `NormalMatrix()` helper.
- Direct TRS matrix construction in `ToMatrix4(Transform)` to avoid building three matrices and multiplying twice.
- Relative-tolerance `NearlyEqual()` for large values.
- Optimized quaternion vector rotation formula.
- SIMD layer:
  - `SIMDVector4f`
  - `SIMDMatrix4f`
  - internal acceleration paths only
- Animation math:
  - `Log`
  - `Exp`
  - `Pow`
  - `Squad`
  - swing-twist decomposition
- `Foundation.Math.Interpolation`:
  - `SmoothStep`
  - `Bezier`
  - `CatmullRom`
  - `MoveTowards`

## Do Not Do Now

- Do not refactor `Vector2`, `Vector3`, and `Vector4` into one generic base type.
- Do not add vector inheritance or virtual dispatch.
- Do not add animation quaternion functions before animation exists.
- Do not keep adding utility functions instead of stress-testing the math layer in real systems.

## Engine Integration Notes

- Vulkan and Metal both benefit from explicit matrix convention docs and `Data()` access.
- CPU ray tracing and BVH work will quickly reveal whether `Ray`, `AABB`, and `Transform` APIs are ergonomic.
- Renderer work will reveal projection, camera, normal-matrix, and transform-order bugs faster than more standalone math polishing.
- Physics work will need stable transforms, inverse transforms, AABBs, planes, and eventually inertia tensors.

## Current Strategic Direction

After the fix-now items and tests:

```text
Stop polishing math.
Build Ray, Plane, AABB.
Then build BVH / CPU Ray Tracer / Camera.
```

Real workloads with thousands of rays, transforms, and bounds will teach more than another round of isolated helper functions.
