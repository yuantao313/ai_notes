# Warp Shuffle

## 设计原因

Warp 内的线程经常需要交换数据。传统做法是用 shared memory：A 线程写、`__syncthreads()`、B 线程读。这需要消耗 shared memory 资源，还要同步，开销不小。

**Warp Shuffle** 让同一个 warp 内的线程直接在寄存器层面交换数据，无需经过 shared memory：

- **无需 `__syncthreads()`** — warp 内的 32 个线程天然是 SIMT 同步的
- **不占 shared memory** — 数据通过寄存器直接传递
- **延迟更低** — 寄存器到寄存器，比 shared memory 读写快

---

## 函数速查

### `__shfl_sync`

```cpp
T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize);
```

把 `srcLane` 线程的 `var` 值广播到同一 warp 的所有参与线程。

```cpp
// lane 0 的值广播给所有 lane（假设 mask = 0xffffffff）
float val = __shfl_sync(0xfffffffff, val, 0);
// → 所有 32 个线程的 val 都等于 lane 0 的那个值
```

### `__shfl_up_sync` / `__shfl_down_sync`

```cpp
T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = warpSize);
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = warpSize);
```

当前 lane 从相邻的 lane 拿数据：

```
// up:  当前 lane 从 lane - delta 取数据
// down: 当前 lane 从 lane + delta 取数据
```

```cpp
// lane 3 从 lane 0 拿：3 - 3 = 0
float val = __shfl_up_sync(0xfffffffff, val, 3);

// lane 0 从 lane 3 拿：0 + 3 = 3
float val = __shfl_down_sync(0xfffffffff, val, 3);
```

### `__shfl_xor_sync`

```cpp
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize);
```

`srcLane = lane ^ laneMask`。XOR 是双射，保证每个目标 lane 有唯一的源 lane。

常用于 Warp 内归约：`laneMask = 16` → `0 ^ 16, 1 ^ 17, ..., 15 ^ 31` 配对交换。

### 参数说明

| 参数 | 含义 |
|---|---|
| `mask` | 参与线程的 bitmask，通常 `0xffffffff`（全 32 线程） |
| `var` | 要交换的值 |
| `srcLane` / `delta` / `laneMask` | 决定从哪个 lane 取数据 |
| `width` | 子 warp 宽度，通常 32（必须 2 的幂，≤ 32） |

> `width` 不为 32 时，warp 被拆成多个子组（如 `width=16` 时两组独立 shuffle）。

### Tips

- **`mask` 参数必须对齐**：比如 `width=16` 时 mask 只作用于子组内的低位 bit
- **`__shfl_down_sync` 最常用于归约**：配合 `offset = 16, 8, 4, 2, 1` 做树形求和
- **不在同一 warp 的线程不能用 shuffle**：跨 warp 通信仍需 shared memory

---

## 编程实战

用 `__shfl_down_sync` 做一个 warp 内的求和（warp-level sum）：

```cpp
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // 所有 lane 都包含总和
}

__global__ void sum_warp(const float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < N) ? x[idx] : 0.0f;

    val = warp_reduce_sum(val);  // warp 内归约

    // lane 0 持有整个 warp 的总和
    if (threadIdx.x % 32 == 0) {
        atomicAdd(y, val);  // 每个 warp 只调一次 atomicAdd
    }
}
```

> `__shfl_down_sync` 的归约过程（以 warp size = 8 为例）：
> ```
> 初始：  [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]
> off=4：  +a4   +a5   +a6   +a7
>         [s0] [s1] [s2] [s3] [a4] [a5] [a6] [a7]
> off=2：  +s2   +s3
>         [t0] [t1] [s2] [s3] [a4] [a5] [a6] [a7]
> off=1：  +t1
>         [sum] [t1] [s2] [s3] [a4] [a5] [a6] [a7]
> ```
> 最终所有 lane 都包含总和，但通常只取 lane 0 的结果。

---

## 硬件实现（扩展）

### PTX 指令

Warp Shuffle 在 PTX 层面对应 `shfl.sync` 系列指令：

```ptx
// __shfl_sync(mask, var, srcLane, width)
shfl.sync.idx.b32  %r0, %r1, %srcLane, %width, %mask;

// __shfl_up_sync(mask, var, delta, width)
shfl.sync.up.b32   %r0, %r1, %delta, %width, %mask;

// __shfl_down_sync(mask, var, delta, width)
shfl.sync.down.b32 %r0, %r1, %delta, %width, %mask;

// __shfl_xor_sync(mask, var, laneMask, width)
shfl.sync.bfly.b32 %r0, %r1, %laneMask, %width, %mask;
```

编译为 SASS 后实际是一条硬件 `SHFL` 指令，由 warp scheduler 直接处理寄存器交叉连接，**不经过任何内存层次**。

### 与 Shared Memory 对比

| | Warp Shuffle | Shared Memory |
|---|---|---|
| 数据路径 | 寄存器 → 寄存器 | 寄存器 → SMEM → 寄存器 |
| 同步 | 无需（warp 内天然同步） | 需要 `__syncthreads()` |
| 延迟 | ~1 cycle | ~20-30 cycles |
| Shared Memory 占用 | 0 | 取决于数据量 |
| 跨 warp 通信 | ❌ 不支持 | ✅ 支持 |

