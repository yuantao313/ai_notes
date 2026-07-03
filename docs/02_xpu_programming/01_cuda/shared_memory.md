# Shared Memory

## 设计原因

每个线程只能直接读写自己的寄存器，但同一个 block 的线程经常需要**共享数据**。

如果不走 shared memory，就只能让每个线程各自从 global memory 读，不仅重复读浪费带宽，还无法协作。

**Shared Memory** 是 SM 内部的一块高速存储，一个 block 内的所有线程都能读写同一份 shared memory，延迟比 global memory 低一个数量级：

```
Global Memory  ←  200~800 cycle  （显存，几百 GB）
        ↓
Shared Memory  ←  20~30 cycle    （SM 内，几十 KB）
        ↓
Register       ←  1 cycle        （线程私有）
```

典型场景：

- **Block 级别**：一个 block 内多个线程需要同一份数据

- **合并访存 + 重用**：从 global memory 读一块到 shared  memory，然后反复读 shared memory

- **线程间通信**：替代 shuffle（跨 warp 时需要 shared memory）

---

## 函数速查

### 声明

```cpp
// 静态 shared memory（编译期确定大小）
__shared__ float cache[256];

// 动态 shared memory（运行时确定大小，在 <<<>>> 第三个参数指定）
extern __shared__ float cache[];     // kernel 参数列表不写，直接在函数体声明
```

启动时指定动态大小：

```cpp
kernel<<<grid, block, shared_mem_size>>>(...);
// shared_mem_size 的单位是字节
```

### 生命周期

| 阶段 | shared memory 状态 |
|---|---|
| kernel 启动 | 分配，内容未初始化 |
| kernel 运行中 | 可读写 |
| kernel 结束 | 释放，不可访问 |

> shared memory 不跨 kernel 持久化。同一个 block 内的线程共享，不同 block 不能访问对方的 shared memory。

### 同步

```cpp
__syncthreads();  // block 内所有线程在此等待，确保 shared memory 写完再读
```

> 如果不加 `__syncthreads()`，一个线程写 shared memory 时，另一个线程可能还没写到，读到的是旧值。

### Bank Conflict

Shared Memory 被分成 32 个 bank（32 位宽）。如果同一 warp 内多个线程访问**同一个 bank 的不同地址**，就会发生 bank conflict，访问被串行化。

```cpp
// 无 conflict：不同线程访问不同 bank
float val = cache[threadIdx.x];      // 线程 i 访问 bank i

// 4-way conflict：4 个线程映射到同一个 bank
float val = cache[threadIdx.x * 4];  // 线程 0→bank 0, 线程 4→bank 0, 8→bank 0...

// 广播：同一个地址无 conflict
float val = cache[0];                // 所有线程读同一地址，硬件广播
```

**常见的 padding 技巧**：在声明时多加一个元素，让相邻行错开 bank，避免 2D 访问时的 conflict：

```cpp
__shared__ float tile[32][32 + 1];   // +1 是为了错开 bank
```

### Tips

- 每个 SM 只有几十 KB（V100: 96 KB, A100: 164 KB, H100: 228 KB），够用但不多
- 静态 + 动态总共不能超过 `cudaDeviceProp.sharedMemPerBlock`
- 动态大小在 `<<<>>>` 的第三个参数指定，单位**字节**
- 读取 shared memory 速度很快，但写完后要 `__syncthreads()` 再让别的线程读

---

## 编程实战

用 shared memory 做 block 内求和：

```cpp
__global__ void sum_shared(const float *x, float *y, int N) {
    __shared__ float cache[256];          // block 内共享缓冲区
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 每个线程从 global memory 读一个元素到 shared memory
    cache[threadIdx.x] = (idx < N) ? x[idx] : 0.0f;
    __syncthreads();                      // 等所有线程写完 cache

    // 2. 树形归约：在 shared memory 上做
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            cache[threadIdx.x] += cache[threadIdx.x + s];
        }
        __syncthreads();                  // 每一步都要同步
    }

    // 3. 第一个线程写回全局
    if (threadIdx.x == 0) {
        atomicAdd(y, cache[0]);
    }
}
```

> 与 Warp Shuffle 的归约对比：shuffle 在寄存器层面做归约，更快但不支持跨 warp；shared memory 支持任意线程数，但需要手动 `__syncthreads()`。实际常用 shuffle 先做 warp 内归约，经过 shared memory 汇总，再做一轮 shuffle。

---

## 硬件实现（扩展）

### PTX

Shared memory 在 PTX 中用 `.shared` 修饰：

```ptx
// 声明 shared memory
.shared .f32 cache[256];

// 读 shared memory → 寄存器
ld.shared.f32 %f0, [cache + %r0];

// 写寄存器 → shared memory
st.shared.f32 [cache + %r0], %f1;
```

对比 global memory 的指令：

```ptx
// global memory
ld.global.f32 %f0, [%rd_addr];   // 200~800 cycle

// shared memory
ld.shared.f32 %f0, [cache + %r0];  // 20~30 cycle
```

### Bank 结构

Shared Memory 硬件上分为 32 个 bank（计算能力 6.x+ 后支持 4 字节宽）：

```
Bank 0: 地址 0,   32,  64,  96, ...   (cache[0], cache[32], ...)
Bank 1: 地址 4,   36,  68, 100, ...   (cache[1], cache[33], ...)
Bank 2: 地址 8,   40,  72, 104, ...   (cache[2], cache[34], ...)
...
Bank 31: 地址 124, 156, 188, 220, ...
```

同一 warp 内：
- **不同 bank** → 并行访问 ✅
- **同一 bank，同一地址** → 广播 ✅
- **同一 bank，不同地址** → Bank Conflict ❌（串行化）
