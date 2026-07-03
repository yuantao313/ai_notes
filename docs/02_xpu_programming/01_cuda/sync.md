# 同步

## Block 内同步

### `__syncthreads()`

block 内所有线程在此等待，全部到齐才继续往下走。

```cpp
__shared__ float cache[256];
cache[threadIdx.x] = data[threadIdx.x];
__syncthreads();                       // 等全部写完再读
float val = cache[threadIdx.x ^ 1];    // 现在可以安全读别人的
```

> ⚠️ 必须 block 内所有线程都能执行到同一个 `__syncthreads()`。如果有 if 分支导致部分线程跳过，会**死锁**。

### `__syncwarp(mask)`

warp 内部分线程等待（SM 6.x+）。`mask` 指定哪些 lane 参与同步。

```cpp
if (lane_id < 16) {
    val = compute();
    __syncwarp(0x0000FFFF);  // 只等低 16 个 lane
    // 其他 16 个 lane 不参与
}
```

> 日常用 `__syncthreads()` 就够了。`__syncwarp` 只在精细控制 warp 内同步时用到。

---

## Thread Fence

不是让线程等待，而是确保**当前线程的访存顺序对外可见**。

| 函数 | 可见范围 |
|---|---|
| `__threadfence_block()` | 同一 block 内 |
| `__threadfence()` | 同一 GPU 上的所有线程 |
| `__threadfence_system()` | 跨 CPU-GPU（SM 6.x+） |

```cpp
// 保证 data 先写完，其他线程看到 flag 时 data 一定就绪
__global__ void producer(int *flag, float *data) {
    data[threadIdx.x] = compute();
    __threadfence();
    atomicExch(flag, 1);          // 标记数据就绪
}
```

> Fence 不是"等别人"，是"让别人看到我干的事"。通常配合 atomic 或 shared memory 一起用。

---

## Host 端同步

| 函数 | 作用 |
|---|---|
| `cudaDeviceSynchronize()` | 等当前设备上所有 work 结束 |
| `cudaStreamSynchronize(stream)` | 等指定 stream 的所有操作完成 |
| `cudaEventSynchronize(event)` | 等某个 event 被记录 |

```cpp
kernel<<<grid, block>>>(...);
cudaDeviceSynchronize();               // 等 kernel 跑完
printf("done\n");

// 或者等单个 stream
cudaStreamSynchronize(my_stream);
```

---

## 常见陷阱

| 场景 | 问题 |
|---|---|
| if 分支里放 `__syncthreads()` | 部分线程跳过 → 死锁 |
| 不加 `__syncthreads()` 就读别人写的 shared memory | 读到旧值 |
| 只改了 shared memory 就以为全局可见 | shared memory 只对 block 内可见 |
| 误认为 warp 内操作不需要考虑同步 | shuffle 不需要，但写 shared memory 后跨 warp 读需要 |

> **一句话总结**：Shared Memory 写完后要 `__syncthreads()` 再读；Global Memory 写完后要让别人读到要加 thread fence。
