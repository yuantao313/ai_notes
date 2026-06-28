# Warp Shuffle

## 相关接口

### __shfl_sync

`T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)`

- srcLane的var按warpSize和mask循环广播到其他lane
- srcLane = srcLane

### __shfl_up_sync/__shfl_down_sync

`T __shfl_up_sync(unsigned mask, T var, int srcLane, int width=warpSize)`

`T __shfl_down_sync(unsigned mask, T var, int srcLane, int width=warpSize)`

- 上下取srcLane的偏移
- srcLane = lane -/+ srcLane

### __shfl_xor_sync

`T __shfl_down_sync(unsigned mask, T var, int laneMask, int width=warpSize)`

- srcLane = lane ^ laneMask
- 直接做xor mask，xor是双射，具有唯一来源

## 使用Warp Shuffle写reducesum

```cpp
__global__ void reduce_shfl(float *x, float*y, const int N){
    __shared__ float data_s[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    float val = (idx < N) ? x[idx] : 0.0f;
    #pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>=1){
        // 直接拿折半规约那个t的内部val
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    // 第一个t回写
    if(lane_id == 0) {
        data_s[warp_id] = val;
    }
    __syncthreads();
    //第一个warp回写
    if(warp_id == 0){
        int warp_num = blockDim.x / warp_size;
        val = (lane_id < warp_num) ? data_s[lane_id] : 0.0f;
        for (int offset = warp_size >> 1; offset > 0; offset >>=1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0){
            atomicAdd(y, val);//一个w调用一次atomicAdd
        }
    }
}
```
