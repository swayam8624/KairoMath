# KairoMath

A high-performance, production-grade 3D math library written in modern C++23. Built entirely on **C++20/C++23 Modules** for lightning-fast compilation, strict encapsulation, and clean symbol isolation.

`KairoMath` is designed to support high-performance game engines, CPU ray tracers, physics solvers, and spatial indexing structures. It follows strict mathematical safety rules, avoids runtime allocations, and maintains complete layout compatibility with GPU graphics APIs (Vulkan, Metal, and OpenGL).

---

## Why KairoMath?

Most C++ math libraries (like GLM) are header-only and rely on massive, legacy preprocessor includes that bloat compile times. `KairoMath` leverages C++20 module partitions, enforcing a clean import boundary:

*   **Lightning Compile Times**: Module interfaces are compiled once into binary module interfaces (`.pcm`), bypassing the need to re-parse headers across translation units.
*   **TRS Transform Paradigm**: Separation of Translation, Rotation (Quaternion), and Scale instead of using generic 4x4 matrices. This prevents floating-point drift, simplifies animation blending/interpolation (via SLerp), and provides clean editor ergonomics.
*   **API Agnostic & Vulkan Ready**: Follows standard **row-major memory storage** (for direct pointer uploads via `.Data()`), **column-vector multiplication** ($v' = M \cdot v$), and **Vulkan-style projection conventions** (Z depth mapped to `[0, 1]`).
*   **No Runtime Allocations**: All objects are trivially copyable, allocation-free, stack-allocated structs designed for cash-friendly cache line access.

---

## Library Architecture

The library is organized in a single linear dependency chain:

```text
Vector (Vec2, Vec3, Vec4)
  └── Matrix (Mat3, Mat4)
        └── Quaternion (Quat)
              └── Transform (TRS)
```

### Module and Namespace Structure
To use the library in your code, import the module partitions under the `Kairo` package:

```cpp
import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.Matrix;
import Kairo.Foundation.Math.Quaternion;
import Kairo.Foundation.Math.Transform;
```

All types, constants, and math functions live inside the C++ namespace:
```cpp
kairo::foundation::math
```

---

## Core Mathematical Operations & Correctness Fixes

`KairoMath` resolves several common mathematical pitfalls present in standard game engine math libraries:

### 1. Snell's Law & Total Internal Reflection (TIR)
In rendering and ray tracing, `Refract()` is used to compute light propagation through media interfaces. The standard Snell formula contains a square root:
$$\cos\theta_2 = \sqrt{1 - \eta^2(1 - \cos^2\theta_1)}$$

If the light travels from a denser to a rarer medium (e.g., glass to air) at a steep angle, the term inside the square root becomes negative. Standard math libraries will evaluate `sqrt(negative)` resulting in `NaN` outputs, which propagates through the scene graph and causes black pixels in renderers.

**Solution**: `KairoMath` guards this condition. If total internal reflection occurs, it immediately handles it by returning a zero-vector sentinel, allowing the caller to cleanly fallback to reflection:
```cpp
const T parallelLengthSquared = T(1) - perpendicular.LengthSquared();
if (parallelLengthSquared < T(0))
{
    return VectorN<T>::Zero(); // Total Internal Reflection guard
}
```

### 2. Arbitrary Axis Projection
Vector projection decomposes motion or forces along an axis. The simplified formula:
$$\text{Project}(u, v) = v \cdot (u \cdot v)$$
is only valid if $v$ is a normalized unit vector.

**Solution**: `KairoMath` implements the generalized projection formula, making it safe for arbitrary non-unit axes, while guarding against divisions by zero (if the axis is zero-length):
```cpp
const T denominator = Dot(onto, onto);
if (denominator <= std::numeric_limits<T>::epsilon())
{
    return VectorN<T>::Zero(); // Safe fallback for zero-length axis
}
return onto * (Dot(vector, onto) / denominator);
```

### 3. Rotated Non-Uniform Scale Inversion
A common bug in scene graphs is attempting to invert a `Transform` with non-uniform scale (e.g., $S = [1, 2, 1]$) that has been rotated. 

Mathematically, the inverse of a rotated non-uniform scale transform introduces **shear**. Because a TRS (Translation, Rotation, Scale) struct has no shear components, the inverse of a rotated non-uniform scale transform **cannot** be represented as another TRS transform. Calling `Inverse()` on a TRS transform with non-uniform scale asserts or produces invalid results.

**Solution**:
*   `Inverse(Transform)` restricts itself to uniform scales (asserting in debug builds when non-uniform scales are detected).
*   For exact point conversions, use `WorldToLocal()`, which handles non-uniform rotated scales exactly by resolving scale divisions on the rotated components:
    ```cpp
    const Vector3<T> translated = point - transform.Translation;
    const Vector3<T> rotated = Rotate(Inverse(transform.Rotation), translated);
    return { rotated.x / transform.Scale.x, rotated.y / transform.Scale.y, ... };
    ```
*   Alternatively, you can convert the transform to a matrix `ToMatrix4(transform)` and compute `Inverse(Matrix4)`, which handles shear exactly in homogeneous coordinates.

### 4. Gauss-Jordan Elimination with Partial Pivoting
For `Inverse(Matrix4)`, `KairoMath` uses Gauss-Jordan elimination with partial pivoting. This approach is highly robust against singular/near-singular matrices (like degenerate camera matrices) and provides numerical stability by finding the largest pivot element in each column to prevent division by near-zero values.

---

## Getting Started: Compilation Guide

Because the library utilizes modern C++20 module interfaces (`.cppm`), you need a compiler and build system that can resolve compile-time module dependencies. On macOS, this requires **CMake 3.28+**, **Ninja**, and **Upstream LLVM Clang** (as Apple's default Xcode Clang does not package the `clang-scan-deps` tool required for scanning modules).

### Prerequisites (macOS)
Install LLVM and Ninja via Homebrew:
```bash
brew install llvm ninja cmake
```

### Build Steps
1.  **Configure CMake** (explicitly targeting the Homebrew LLVM clang++ binary):
    ```bash
    cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
    ```
2.  **Compile the Targets**:
    ```bash
    cmake --build build
    ```
3.  **Run the Tests**:
    ```bash
    ./build/MathTests
    ```
4.  **Run the Visualizer**:
    ```bash
    ./build/MathVisualizer
    ```

---

## Verifying Visual Operations

`KairoMath` comes with an interactive visual test suite that lets you visualize math operations directly.

### 1. Terminal 3D Render
Running `./build/MathVisualizer` displays a real-time, 60-frame interactive 3D rotating wireframe cube directly inside your terminal, calculated using `KairoMath`'s `Transform`, `Quaternion`, and `Perspective` projection matrices.

### 2. Browser Interactive Playground
Running the visualizer also generates an HTML file `visual_tests.html` in your directory. Open it on your Macbook:
```bash
open visual_tests.html
```
This launches a beautiful, modern, dark-mode dashboard where you can interactively adjust:
*   **Vector Projection**: Drag target vectors and axes to observe projection behavior on non-unit and zero-length boundaries.
*   **Reflection & Refraction**: Tweak refractive indices to see Snell's law in action and watch the refracted ray disappear during **Total Internal Reflection**.
*   **3D Transform Workspace**: Modify Translation, Rotation, and Scale sliders to inspect the resulting homogeneous $4\times 4$ TRS matrix composition in real-time.

---

## Code Usage Tutorial

Here is a quick reference guide on how to program with `KairoMath`:

### 1. Vector Operations
```cpp
import Kairo.Foundation.Math.Vector;
using namespace kairo::foundation::math;

// Construct vectors
Vec3f a(1.0f, 2.0f, 3.0f);
Vec3f b = Vec3f::Up(); // [0, 1, 0]

// Math functions
float dotProduct = Dot(a, b);
Vec3f crossProduct = Cross(a, b);
Vec3f normalized = a.Normalized();
```

### 2. Rotations with Quaternions
```cpp
import Kairo.Foundation.Math.Quaternion;
using namespace kairo::foundation::math;

// Create rotation of 90 degrees around Y axis
float angle = 3.14159265f / 2.0f;
Quatf rotation = RotationAroundY(angle);

// Get direction vectors
Vec3f forwardDir = Forward(rotation);
Vec3f rightDir = Right(rotation);
```

### 3. Transforms (TRS)
```cpp
import Kairo.Foundation.Math.Transform;
using namespace kairo::foundation::math;

// Setup transform hierarchy
Transformf parent(Vec3f(0.0f, 5.0f, 0.0f), RotationAroundY(0.5f), Vec3f(1.0f, 1.0f, 1.0f));
Transformf child(Vec3f(2.0f, 0.0f, 0.0f));

// Compose transforms (applies child, then parent)
Transformf combined = parent * child;

// Transform points from local to world space
Vec3f localPoint(1.0f, 0.0f, 0.0f);
Vec3f worldPoint = TransformPoint(combined, localPoint);

// Transform back to local space (handles rotated non-uniform scale safely)
Vec3f restoredPoint = WorldToLocal(combined, worldPoint);
```

---

## Future Roadmap

The strategic direction of KairoMath is to remain focused on geometry primitives next before adding camera/frustum/physics wrappers:

*   [ ] Add `Ray.cppm`, `Plane.cppm`, and `AABB.cppm` targets.
*   [ ] Implement BVH (Bounding Volume Hierarchy) construction.
*   [ ] Implement a CPU Ray Tracer to stress-test vector/matrix operations.
*   [ ] Add relative-tolerance comparisons for very large values.
