# 向量化内存访问

## 设计原因

GPU 的显存带宽很高（H100 超过 3 TB/s），但要充分利用带宽，访问模式必须满足**合并访问（coalesced access）**规则——一个 warp 32 个线程的访存请求，如果能合并为尽量少的内存事务，带宽利用率就高。

以 `float` 为例，每个线程读 1 个 float（4 字节），32 个线程共 128 字节。如果地址连续，硬件可以合并为一次 128 字节事务；但如果地址分散，就会拆成多次小事务，带宽利用率大幅下降。

**向量化**的思想很简单：一个线程一次读 4 个 float（128 位），整个 warp 一次读 `32 × 128 = 4096` 字节。虽然总数据量不变，但 **`float4` 告诉编译器用 `LDG.128` 指令**，减少指令数和地址计算开销，同时天然保证 16 字节对齐，更容易触发合并访问。

---

## 函数速查

### float4 基本操作

```cpp
#define FLOAT4(ptr) (reinterpret_cast<float4&>(ptr))

// float4 读：把连续 4 个 float 强转为 float4
float a[4] = {1, 2, 3, 4};
float4 v = FLOAT4(a[0]);   // v.x=1, v.y=2, v.z=3, v.w=4

// float4 写回
FLOAT4(a[0]) = make_float4(5, 6, 7, 8);
```

### 向量类型一览

| 类型 | 宽度 | 分量 | 内存事务 |
|---|---|---|---|
| `float`（标量）| 32-bit | — | 1 次 `LDG.32` |
| `float2` | 64-bit | `.x, .y` | 1 次 `LDG.64` |
| `float4` | 128-bit | `.x, .y, .z, .w` | 1 次 `LDG.128` |
| `double2` | 128-bit | `.x, .y` | 1 次 `LDG.128` |
| `int4` / `uint4` | 128-bit | `.x, .y, .z, .w` | 1 次 `LDG.128` |
| `half2` | 32-bit | `.x, .y` | 1 次 `LDG.32` |

### Tips

- **16 字节对齐**：`cudaMalloc` 默认 256 字节对齐，没问题；若用 `cudaMallocManaged` 或偏址指针需注意
- **Grid 减为 1/4**：每个线程处理 4 个元素，总线程数 = `ceil(N / 4)`
- **尾数处理**：N 非 4 倍数时，主体用 float4，尾部用标量补全
- **适用场景**：Element-wise 运算（add / mul / scale / 激活函数），连续地址访问收益最大

---

## 编程实战

用 float4 写一个向量加法：

```cpp
#define FLOAT4(ptr) (reinterpret_cast<float4&>(ptr))

__global__ void add_vectorized(float *x, float *y, float *z, const int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // 每个线程处理 4 个元素

    // 主体：一次 float4 读写，保证不越界
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
        // 尾部：标量补全余下的 0~3 个元素
        for (int i = idx; i < N; i++) {
            z[i] = x[i] + y[i];
        }
    }
}
```

> 尾部处理不能少——否则 N 不是 4 倍数时会越界。

---

## 硬件实现（扩展）

### PTX 对比：标量 vs 向量化

编译后，标量版本每条访存编译为 `LDG.32` / `STG.32`：

```ptx
// 标量版本：每个线程读写 1 个 float → LDG.32
ld.global.f32  %f1, [%rd_x];      // 读 x[idx]
ld.global.f32  %f2, [%rd_y];      // 读 y[idx]
add.f32        %f3, %f1, %f2;     // 加法
st.global.f32  [%rd_z], %f3;      // 写 z[idx]
// 共 3 条访存指令（2 条 LDG.32 + 1 条 STG.32）
```

float4 版本编译为 `LDG.128` / `STG.128`：

```ptx
// float4 版本：每个线程读写 4 个 float → LDG.128
ld.global.v4.f32  {%f1, %f2, %f3, %f4}, [%rd_x];  // 一次读 4 个 float
ld.global.v4.f32  {%f5, %f6, %f7, %f8}, [%rd_y];
add.f32  %f9,  %f1, %f5;     // 4 次加法
add.f32  %f10, %f2, %f6;
add.f32  %f11, %f3, %f7;
add.f32  %f12, %f4, %f8;
st.global.v4.f32  [%rd_z], {%f9, %f10, %f11, %f12};  // 一次写 4 个 float
// 共 3 条访存指令（2 条 LDG.128 + 1 条 STG.128），但每条约 4 倍数据量
```

关键区别：

| | 标量 | float4 |
|---|---|---|
| PTX 指令 | `ld.global.f32` | `ld.global.v4.f32` |
| SASS 指令 | `LDG.32` | `LDG.128` |
| 每线程数据量 | 4 字节 | 16 字节 |
| 指令数（访存） | 3 × threads | 3 × threads（但每指令 4× 数据） |

> 虽然指令数量不变，但 `LDG.128` 让内存控制器每次处理 128 位宽度的请求，**每个线程的地址计算和请求调度开销被分摊到 4 个元素上**，对带宽敏感的 kernel 效果显著。
