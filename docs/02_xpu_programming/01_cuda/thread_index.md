# 线程索引与层级

## 速查

### 内置变量

| 变量 | 含义 | 例 | PTX |
|---|---|---|---|
| `threadIdx` | 当前线程在 block 里的编号 | `threadIdx.x = 0, 1, 2, ...` | `%tid.x` |
| `blockIdx` | 当前 block 在 grid 里的编号 | `blockIdx.x = 0, 1, 2, ...` | `%ctaid.x` |
| `blockDim` | 一个 block 有多少线程 | `blockDim.x = 256` | `%ntid.x` |
| `gridDim` | 一个 grid 有多少 block | `gridDim.x = 100` | `%nctaid.x` |
| `warpSize` | 一个 warp 有多少线程 | 永远 `32` | 常量 |

### 常用公式

| 想要什么 | 怎么写 |
|---|---|
| 全局唯一 ID（1D） | `blockIdx.x * blockDim.x + threadIdx.x` |
| 当前线程在 warp 里的位置（lane） | `threadIdx.x % 32` |
| 当前线程在 block 里的 warp 编号 | `threadIdx.x / 32` |
| block 数量 | `(N + blockDim.x - 1) / blockDim.x` |

### 启动 kernel 时

```cpp
// N 个元素，每个 block 256 线程
int grid = (N + 255) / 256;           // 向上取整
kernel<<<grid, 256>>>(...);

// kernel 内部
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {                         // 边界检查
    // 干活
}
```

### 2D 启动

```cpp
dim3 block(16, 16);
dim3 grid((W + 15) / 16, (H + 15) / 16);
kernel<<<grid, block>>>(...);

// kernel 内部
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * W + x;  // 展平为一维
```

---

## 结构

```
grid                   ← 一次 kernel 启动
 └── block 0           ← 分到同一个 SM
      ├── warp 0       ← 32 个线程一起执行
      │    ├── lane 0
      │    ├── lane 1
      │    └── ...
      ├── warp 1
      └── ...
```

- **Warp**：GPU 真正干活的最小单位，32 个线程绑定在一起执行同一条指令
- **Lane**：线程在 warp 里的编号（0~31），相当于座位号

---

## Tips

- `blockDim.x` 最好设成 32 的倍数（比如 128 / 256 / 512），避免最后一个 warp 填不满浪费
- kernel 里一定要 `if (idx < N)`，因为 grid 算出来可能多出几个线程
- 同一个 warp 里的线程天然同步，不用 `__syncthreads()`
- 不同 warp 之间想同步才需要 `__syncthreads()`
