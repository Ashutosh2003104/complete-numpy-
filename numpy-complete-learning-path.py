# Comprehensive NumPy Learning Path

## Phase 1: NumPy Basics and Array Creation

```python
import numpy as np

# 1.1 Array Creation Methods
# Creating basic arrays
list_array = np.array([1, 2, 3, 4, 5])
matrix_array = np.array([[1, 2, 3], [4, 5, 6]])

# Special array creation
zeros_array = np.zeros(5)
ones_array = np.ones((2, 3))
range_array = np.arange(0, 10, 2)
linspace_array = np.linspace(0, 1, 5)

# Array Attributes
print("Array Shape:", matrix_array.shape)
print("Array Dimensions:", matrix_array.ndim)
print("Data Type:", matrix_array.dtype)
print("Array Size:", matrix_array.size)
```

## Phase 2: Array Indexing and Basic Operations

```python
# 2.1 Basic Indexing
arr_1d = np.array([10, 20, 30, 40, 50])
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 1D Indexing
print("First element:", arr_1d[0])
print("Last element:", arr_1d[-1])

# 2D Indexing
print("First row:", arr_2d[0])
print("Specific element:", arr_2d[1, 2])

# Slicing
print("First three elements:", arr_1d[:3])
print("Every second element:", arr_1d[::2])

# Mathematical Operations
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([10, 20, 30, 40])

print("Addition:", arr1 + arr2)
print("Multiplication:", arr1 * arr2)
print("Square root:", np.sqrt(arr1))
```

## Phase 3: Array Manipulation and Reshaping

```python
# 3.1 Reshaping Arrays
arr = np.arange(12)
reshaped_2d = arr.reshape(3, 4)
reshaped_3d = arr.reshape(2, 3, 2)

# Concatenation
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
horizontal_concat = np.concatenate([arr1, arr2])

arr_2d1 = np.array([[1, 2], [3, 4]])
arr_2d2 = np.array([[5, 6], [7, 8]])
vertical_concat = np.concatenate([arr_2d1, arr_2d2], axis=0)

# Stacking
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
vertical_stack = np.vstack((x, y))
horizontal_stack = np.hstack((x, y))
```

## Phase 4: Boolean Indexing and Conditional Operations

```python
# 4.1 Boolean Indexing
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
greater_than_5 = arr > 5
print("Elements > 5:", arr[greater_than_5])

# Conditional Selection
replaced = np.where(arr < 3, arr * 2, arr)

# Multiple Conditions
conditions = [arr < 3, arr >= 3]
choices = [arr * 2, arr + 10]
result = np.select(conditions, choices)
```

## Phase 5: Statistical Operations and Random Generation

```python
# 5.1 Statistical Functions
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print("Mean:", np.mean(arr))
print("Median:", np.median(arr))
print("Standard Deviation:", np.std(arr))

# Random Number Generation
np.random.seed(42)
uniform_dist = np.random.uniform(0, 1, 5)
normal_dist = np.random.normal(0, 1, 5)
random_ints = np.random.randint(1, 10, 5)

# Percentiles
arr_stats = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("25th Percentile:", np.percentile(arr_stats, 25))
```

## Phase 6: Linear Algebra and Advanced Math

```python
# 6.1 Matrix Operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix Multiplication
matrix_mult = np.dot(A, B)
print("Matrix Multiplication:\n", matrix_mult)

# Linear Algebra Functions
eigenvalues, eigenvectors = np.linalg.eig(A)
inverse = np.linalg.inv(A)
det = np.linalg.det(A)

# Advanced Mathematical Operations
x = np.linspace(0, np.pi, 5)
print("Sine:", np.sin(x))
print("Exponential:", np.exp(x))
```

## Phase 7: Broadcasting

```python
# 7.1 Broadcasting
arr = np.array([1, 2, 3, 4])
print("Scalar Multiplication:", arr * 2)

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_1d = np.array([10, 20, 30])
print("Broadcasting:\n", arr_2d + arr_1d)

# Dimension Manipulation
col_vector = arr[:, np.newaxis]
row_vector = arr[np.newaxis, :]
```

## Phase 8: File I/O and Data Processing

```python
# 8.1 Saving and Loading Arrays
arr = np.array([1, 2, 3, 4, 5])
np.save('my_array.npy', arr)
loaded_arr = np.load('my_array.npy')

# Multiple Array Saving
multi_arr = [np.array([1,2,3]), np.array([4,5,6])]
np.savez('multiple_arrays.npz', arr1=multi_arr[0], arr2=multi_arr[1])

# Text File Handling
data = np.array([[1, 2, 3], [4, 5, 6]])
np.savetxt('data.csv', data, delimiter=',', fmt='%d')
loaded_text = np.loadtxt('data.csv', delimiter=',')
```

## Phase 9: Advanced NumPy Techniques

```python
# 9.1 Memory-Efficient Operations
arr = np.arange(1_000_000)
view = arr[::2]  # Memory view
copy = arr[::2].copy()  # Explicit copy

# Structured Arrays
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]
data = np.array([
    ('Alice', 25, 1.65),
    ('Bob', 30, 1.80)
], dtype=dtype)

# Performance Optimization
import timeit
x = np.linspace(0, np.pi, 1000)

def vectorized_sin(x):
    return np.sin(x)

def loop_sin(x):
    return [np.sin(val) for val in x]

print("Vectorized Time:", timeit.timeit(lambda: vectorized_sin(x), number=1000))
print("Loop Time:", timeit.timeit(lambda: loop_sin(x), number=1000))
```

## Learning Milestones
1. Array Creation and Manipulation
2. Mathematical Operations
3. Statistical Analysis
4. Linear Algebra
5. File Handling
6. Advanced Techniques
