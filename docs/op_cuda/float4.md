# float4 向量化内存访问

CUDA 中用 `float4` 一次读写 4 个 float（128-bit），合并为一次内存事务，提升带宽利用率。

## 基本用法

`float4` 的内存在 `float4.x / .y / .z / .w` 四个分量，和普通 float 数组通过 `reinterpret_cast` 互相转换。

```cpp
#define FLOAT4(ptr) (reinterpret_cast<float4&>(ptr))

// 把连续的 float 数组强转为 float4 读
float a[4] = {1, 2, 3, 4};
float4 v = FLOAT4(a[0]);   // v.x=1, v.y=2, v.z=3, v.w=4

// float4 写回 float 数组
FLOAT4(a[0]) = make_float4(5, 6, 7, 8);
```

## 对比示例：逐元素 vs 向量化

### 标量版本（一次处理 1 个 float）

```cpp
__global__ void add_scalar(float *x, float *y, float *z, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        z[idx] = x[idx] + y[idx];
    }
}
```

### float4 向量化版本（一次处理 4 个 float）

```cpp
#define FLOAT4(ptr) (reinterpret_cast<float4&>(ptr))

__global__ void add_float4(float *x, float *y, float *z, const int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // 每个线程处理 4 个元素
    if (idx < N) {
        float4 x_tmp = FLOAT4(x[idx]);
        float4 y_tmp = FLOAT4(y[idx]);
        float4 z_tmp;
        z_tmp.x = x_tmp.x + y_tmp.x;
        z_tmp.y = x_tmp.y + y_tmp.y;
        z_tmp.z = x_tmp.z + y_tmp.z;
        z_tmp.w = x_tmp.w + y_tmp.w;
        FLOAT4(z[idx]) = z_tmp;
    }
}
```

### 处理 N 非 4 倍数的场景

主体用 float4 跑 4 的倍数部分，尾部 1~3 个用标量补全。

```cpp
#define FLOAT4(ptr) (reinterpret_cast<float4&>(ptr))

__global__ void add_float4_tail(float *x, float *y, float *z, const int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // 主体：一次处理 4 个，保证不越界
    if (idx + 3 < N) {
        float4 x_tmp = FLOAT4(x[idx]);
        float4 y_tmp = FLOAT4(y[idx]);
        float4 z_tmp;
        z_tmp.x = x_tmp.x + y_tmp.x;
        z_tmp.y = x_tmp.y + y_tmp.y;
        z_tmp.z = x_tmp.z + y_tmp.z;
        z_tmp.w = x_tmp.w + y_tmp.w;
        FLOAT4(z[idx]) = z_tmp;
    } else {
        // 尾部：标量逐元素处理余下的 0~3 个
        for (int i = idx; i < N; i++) {
            z[i] = x[i] + y[i];
        }
    }
}
```

这样无论 N 是多少都能正确运行，Grid 仍按 `ceil(N/4)` 计算。

## 注意事项

| 要点 | 说明 |
|---|---|
| **Grid 减为 1/4** | 每个线程处理 4 个元素，总线程数 = `ceil(N/4)` |
| **16 字节对齐** | 数组首地址需 `cudaMalloc` 分配（自动对齐），或 `__align__(16)` |
| **适用场景** | Element-wise 运算（add / mul / scale），相邻线程访问连续地址时收益最大 |

## 其他向量类型

| 类型 | 宽度 | 分量 |
|---|---|---|
| `float2` | 64-bit | `.x`, `.y` |
| `float4` | 128-bit | `.x`, `.y`, `.z`, `.w` |
| `double2` | 128-bit | `.x`, `.y` |
| `int4` / `uint4` | 128-bit | `.x`, `.y`, `.z`, `.w` |
| `half2` | 32-bit | `.x`, `.y` |
