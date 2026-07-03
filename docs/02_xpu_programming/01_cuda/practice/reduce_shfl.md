# Block 级 Warp Shuffle 归约

综合运用 Warp Shuffle 和 Shared Memory 实现的 block 级归约 kernel：

```cpp
__global__ void reduce_shfl(float *x, float *y, const int N) {
    __shared__ float data_s[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    float val = (idx < N) ? x[idx] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane_id == 0) data_s[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        int warp_num = blockDim.x / 32;
        val = (lane_id < warp_num) ? data_s[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) atomicAdd(y, val);
    }
}
```

## 工作流程

1. **Warp 内归约**：每个 warp 用 `__shfl_down_sync` 做树形求和，32 → 1
2. **写 Shared Memory**：每个 warp 的 lane 0 把结果写到 `data_s[warp_id]`
3. **同步**：`__syncthreads()` 确保所有 warp 写完
4. **第二个 Warp 归约**：warp 0 从 shared memory 读回所有 warp 的部分和，再次 shuffle 归约
5. **写回全局**：lane 0 用 `atomicAdd` 输出到 `y`
