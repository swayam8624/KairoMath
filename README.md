# KairoMath

A high-performance, production-grade 3D math and linear algebra library written in modern C++23. Built entirely on **C++20/C++23 Modules** for lightning-fast compilation, strict encapsulation, and clean symbol isolation.

`KairoMath` is designed to support high-performance game engines, CPU ray tracers, physics solvers, spatial indexing structures, and data science/robotics applications. It follows strict mathematical safety rules, avoids unnecessary runtime allocations, and maintains complete layout compatibility with GPU graphics APIs.

---

## Why KairoMath?

Most C++ math libraries (like GLM) are header-only and rely on massive, legacy preprocessor includes that bloat compile times. `KairoMath` leverages C++20 module partitions, enforcing a clean import boundary:

*   **Lightning Compile Times**: Module interfaces are compiled once into binary module interfaces (`.pcm`), bypassing the need to re-parse headers across translation units.
*   **TRS Transform Paradigm**: Separation of Translation, Rotation (Quaternion), and Scale instead of using generic 4x4 matrices. This prevents floating-point drift, simplifies animation blending/interpolation (via SLerp), and provides clean editor ergonomics.
*   **API Agnostic & Vulkan Ready**: Follows standard **row-major memory storage** (for direct pointer uploads via `.Data()`), **column-vector multiplication** ($v' = M \cdot v$), and **Vulkan-style projection conventions** (Z depth mapped to `[0, 1]`).
*   **Linear Algebra Suite**: Built-in support for dynamically sized matrices, linear system solvers, matrix decompositions, eigenvalue solvers, SVD, PCA, and regression.
*   **No Runtime Allocations for Core Geometry**: Geometry types (Vec/Mat/Quat) are trivially copyable, allocation-free, stack-allocated structs designed for cash-friendly cache line access.

---

## Library Architecture

The library is organized in a clean hierarchical dependency chain of module partitions:

### 1. Geometry & Spatial Primitives
```text
Vector (Vec2, Vec3, Vec4)
  └── Matrix (Mat2, Mat3, Mat4)
        └── Quaternion (Quat)
              └── Transform (TRS)
```
*   **Import Interfaces**:
    ```cpp
    import Kairo.Foundation.Math.Vector;
    import Kairo.Foundation.Math.Matrix;
    import Kairo.Foundation.Math.Quaternion;
    import Kairo.Foundation.Math.Transform;
    ```

### 2. Linear Algebra & Numerical Suite
```text
DynamicMatrix (arbitrary dimensions)
  ├── LinearSolve (Forward/Backward Substitution, Gauss-Jordan, REF/RREF)
  ├── Decomposition (LU, LUP, QR, Cholesky, LDLT)
  ├── Eigen (Power Iteration, QR Eigenvalues)
  ├── SVD (Singular Value Decomposition)
  └── Statistics (Covariance, PCA, Linear Regression)
```
*   **Import Interfaces**:
    ```cpp
    import Kairo.Foundation.Math.DynamicMatrix;
    import Kairo.Foundation.Math.LinearAlgebra.LinearSolve;
    import Kairo.Foundation.Math.LinearAlgebra.Decomposition;
    import Kairo.Foundation.Math.LinearAlgebra.Eigen;
    import Kairo.Foundation.Math.LinearAlgebra.SVD;
    import Kairo.Foundation.Math.LinearAlgebra.Statistics;
    ```

All types, constants, and math functions live inside the C++ namespace:
```cpp
kairo::foundation::math
```

### Umbrella Import

Use the umbrella module when a translation unit needs the full math surface:

```cpp
import Kairo.Foundation.Math;
```

Use narrower module imports when you want tighter compile dependencies:

```cpp
import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.Matrix;
import Kairo.Foundation.Math.Quaternion;
import Kairo.Foundation.Math.Transform;
import Kairo.Foundation.Math.DynamicMatrix;
import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.LinearAlgebra.LinearSolve;
import Kairo.Foundation.Math.LinearAlgebra.Decomposition;
import Kairo.Foundation.Math.LinearAlgebra.Eigen;
import Kairo.Foundation.Math.LinearAlgebra.SVD;
import Kairo.Foundation.Math.LinearAlgebra.Statistics;
```

### Verified Foundation Status

The current module surface is complete for the Phase A math foundation:
`Vector`, `Matrix`, `Quaternion`, `Transform`, `DynamicMatrix`, `Tensor`,
`LinearSolve`, `Decomposition`, `Eigen`, `SVD`, and `Statistics` are exported by
the umbrella module and included in the CMake module file set.

---

## Core Features & Stabilization Fixes

`KairoMath` resolves several common mathematical pitfalls present in standard game engine math libraries:

### 1. Snell's Law & Total Internal Reflection (TIR)
In rendering and ray tracing, `Refract()` is used to compute light propagation through media interfaces.
If the light travels from a denser to a rarer medium (e.g., glass to air) at a steep angle, standard math libraries will evaluate `sqrt(negative)` resulting in `NaN` outputs, which propagates through the scene graph and causes black pixels in renderers.

**Solution**: `KairoMath` guards this condition. If total internal reflection occurs, it immediately handles it by returning a zero-vector sentinel, allowing the caller to cleanly fallback to reflection.

### 2. Arbitrary Axis Projection
Vector projection decomposes motion or forces along an axis. The simplified formula is only valid if the axis is a normalized unit vector.

**Solution**: `KairoMath` implements the generalized projection formula, making it safe for arbitrary non-unit axes, while guarding against divisions by zero (if the axis is zero-length).

### 3. Rotated Non-Uniform Scale Inversion
The inverse of a rotated non-uniform scale transform introduces **shear**. Because a TRS struct has no shear components, the inverse of a rotated non-uniform scale transform **cannot** be represented as another TRS transform.

**Solution**:
*   `Inverse(Transform)` restricts itself to uniform scales (asserting in debug builds when non-uniform scales are detected).
*   For exact point conversions, use `WorldToLocal()`, which handles non-uniform rotated scales exactly by resolving scale divisions on the rotated components.
*   Alternatively, convert the transform to a matrix `ToMatrix4(transform)` and compute `Inverse(Matrix4)`.

### 4. Robust Gauss-Jordan Elimination with Partial Pivoting
For `Inverse(Matrix4)` and general linear solver functions, `KairoMath` uses Gauss-Jordan elimination with partial pivoting. This approach is highly robust against singular/near-singular matrices and provides numerical stability by finding the largest pivot element in each column to prevent division by near-zero values.

### 5. Production-Grade Matrix Operations (LUP-Based)
For arbitrary dynamically-sized matrices (`DynamicMatrix`), `KairoMath` provides LUP-decomposition-based implementations of key linear algebra operations:
*   **`Determinant`**: Computes the determinant using LUP decomposition, tracking the sign of the permutation vector via disjoint cycle decomposition, and multiplying diagonal entries of $U$.
*   **`Inverse`**: Inverts a square matrix by performing forward/backward substitution on LUP factors column-by-column against the identity matrix.
*   **`Rank`**: Computes the 2-norm rank using Singular Value Decomposition with a robust tolerance scale.
*   **`ConditionNumber`**: Computes the 2-norm condition number ($\sigma_{\max} / \sigma_{\min}$) utilizing the singular values from SVD, returning infinity for singular matrices.

### 6. Upgraded Eigen Solver (Householder Reduction & Tridiagonalization)
Symmetric eigenvalues and eigenvectors are computed using the standard production pipeline:
1.  **Householder Reduction**: Reduces the symmetric matrix to symmetric tridiagonal form ($T = Q^T A Q$) in one pass of Householder reflections.
2.  **Implicit QR Sweeps**: Runs the implicitly shifted symmetric QR algorithm with Wilkinson shifts directly on $T$. This reduces the complexity per iteration from $O(n^3)$ to $O(n)$.

### 7. Upgraded SVD Solver (Golub-Kahan Bidiagonal SVD)
Thin Singular Value Decomposition is computed using bidiagonalization:
1.  **Golub-Kahan Bidiagonalization**: Alternates left and right Householder reflections to reduce a general rectangular matrix to upper bidiagonal form ($B = U_{bid}^T A V_{bid}$).
2.  **Bidiagonal SVD**: Computes the SVD of $B$ by solving the eigenvalues of the symmetric tridiagonal matrix $T = B^T B$ using the fast tridiagonal QR algorithm, and then accumulates the left and right singular vectors.

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
4.  **Run the Visualizer & Generate HTML Report**:
    ```bash
    ./build/MathVisualizer
    ```

---

## Interactive Visual Lab & API Server

`KairoMath` includes a local python gateway server (`server.py`) and standard HTML visualizer to interactively test and verify all mathematical algorithms with the C++ backend.

### Architecture

```text
  [ Next.js Portal / Web Dashboard ] (Browser)
                 │  (REST API / JSON)
                 ▼
      [ Python API Server ] (visual/server.py)
                 │  (std::cin / std::cout)
                 ▼
       [ C++ Math Engine ] (MathVisualizer --api)
```

1. **C++ API Mode**: The `MathVisualizer` binary accepts `--api` which reads mathematical payloads from `stdin`, executes calculations, and prints a single-line JSON result to `stdout`.
2. **Python API Gateway**: The script `visual/server.py` listens on `http://localhost:8080`. It routes `/api/<endpoint>` requests directly to a spawned C++ `MathVisualizer --api` subprocess, enabling web dashboards to run calculations on the native C++ library.
3. **Legacy HTML Visualizer**: Located at `visual/visual_tests.html`, this single-page dashboard can be served statically.

### Running the Live Visual Lab

1. **Ensure the C++ Visualizer is compiled**:
   ```bash
   cmake --build build
   ```
2. **Start the Python gateway server**:
   ```bash
   python3 visual/server.py
   ```
3. **Access the Legacy Dashboard**:
   Open [http://localhost:8080/visual_tests.html](http://localhost:8080/visual_tests.html) in your browser.

---

## Premium Documentation Portal (Next.js + Shadcn UI)

A next-generation developer documentation portal is located in the `Kairo/docs` directory. It is built using **Next.js (App Router)**, **TypeScript**, **Tailwind CSS v4**, and **Shadcn UI**.

### Features

* **Structured Interactive Guides**: Detailed compiler configurations, CMake setup, and modular reference docs for Vectors, Quaternions, Linear Solvers, and Applied Statistics.
* **Interactive Playgrounds**:
  - *Vector Algebra Lab*: Vector projection & Snell refraction with Total Internal Reflection boundary visualizer on SVG.
  - *TRS Transform Lab*: 3D wireframe cube renderer showing real-time 4x4 transform matrices.
  - *Applied Statistics Lab*: Add 2D data points to compute regression lines and Principal Component Analysis (PCA) axes.
  - *General Matrix Sandbox*: Run Determinants, Inverses, Rank, Condition Numbers, LUP decompositions, SVD, and QR sweeps on 2x2 to 6x6 matrices.
* **Hybrid Execution Engine**: Automatically routes computations to the native C++ engine (port `8080`) when online, and defaults to type-safe client-side JavaScript calculations when offline.

### Running the Documentation Portal

1. **Launch the C++ API Server**:
   ```bash
   python3 Foundation/KairoMath/visual/server.py
   ```
2. **Launch the Next.js Dev Server**:
   ```bash
   cd docs
   npm run dev
   ```
3. Open [http://localhost:3000](http://localhost:3000) in your browser.

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

### 4. Solving Linear Systems and Decompositions
```cpp
import Kairo.Foundation.Math.DynamicMatrix;
import Kairo.Foundation.Math.LinearAlgebra.LinearSolve;
import Kairo.Foundation.Math.LinearAlgebra.Decomposition;

using namespace kairo::foundation::math;

// Create a 3x3 matrix and vector b
DynamicMatrix<double> A(3, 3);
A(0,0)=2.0; A(0,1)=1.0; A(0,2)=1.0;
A(1,0)=1.0; A(1,1)=3.0; A(1,2)=1.0;
A(2,0)=1.0; A(2,1)=1.0; A(2,2)=4.0;

std::vector<double> b = {2.0, 3.0, 4.0};

// Solve system A x = b
std::vector<double> x = LinearSolve(A, b);

// Compute LUP Factorization
LUPResult<double> res = LUP(A);
// res.L, res.U, and res.P represent the components
```
