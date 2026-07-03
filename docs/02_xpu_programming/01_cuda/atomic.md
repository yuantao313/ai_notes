# 原子函数

## 设计原因

为什么需要原子操作？因为普通的读-改-写在多 core 并行时会出现 race condition。

假设两个核心同时给 `[addr]` 加 1，初始值 `10`，预期结果 `12`：

```ptx
// 非原子的读-改-写（直觉上的三条指令）
ld.global.u32 %r1, [%rd_addr];  // 1. 读
add.u32 %r1, %r1, 1;            // 2. 改
st.global.u32 [%rd_addr], %r1;  // 3. 写
```

如果两个 core **顺序执行**，结果正确：
```ptx
ld.global.u32 %r1, [%rd_addr];  // Core 1  %r1 = 10
add.u32 %r1, %r1, 1;            // Core 1  %r1 = 11
st.global.u32 [%rd_addr], %r1;  // Core 1  [addr] = 11
ld.global.u32 %r1, [%rd_addr];  // Core 2  %r1 = 11
add.u32 %r1, %r1, 1;            // Core 2  %r1 = 12
st.global.u32 [%rd_addr], %r1;  // Core 2  [addr] = 12  ✓
```

但实际运行时指令可能**交错执行**，结果错误：
```ptx
ld.global.u32 %r1, [%rd_addr];  // Core 1  %r1 = 10
ld.global.u32 %r1, [%rd_addr];  // Core 2  %r1 = 10    ← 同时读到旧值
add.u32 %r1, %r1, 1;            // Core 1  %r1 = 11
add.u32 %r1, %r1, 1;            // Core 2  %r1 = 11
st.global.u32 [%rd_addr], %r1;  // Core 1  [addr] = 11
st.global.u32 [%rd_addr], %r1;  // Core 2  [addr] = 11  ✗ 少加了一次！
```

两个 core 同时读到了 `10`，各自 +1 写回 `11`，一次加法被吞掉了。

**原子操作** 把读-改-写合并为一条不可分割的指令，从根源上解决这个问题。

---

## 函数速查

### Arithmetic（算数运算）

| API | T | Func |
|---|---|---|
| `template<class T> T atomicAdd(T* address, T val)` | `i32, u32, u64, f32, f64(6.x+), __half/nv_bfloat16(8.x+)` | `old = *address; *address += val; return old` |
| `template<class T> T atomicSub(T* address, T val)` | `i32, u32` | `old = *address; *address -= val; return old` |
| `template<class T> T atomicExch(T* address, T val)` | `i32, u32, u64, f32, T*` | `old = *address; *address = val; return old` |
| `template<class T> T atomicMin(T* address, T val)` | `i32, u32, u64, i64` | `old = *address; *address = min(old, val); return old` |
| `template<class T> T atomicMax(T* address, T val)` | `i32, u32, u64, i64` | `old = *address; *address = max(old, val); return old` |
| `unsigned atomicInc(unsigned* address, unsigned val)` | `u32` | `old = *address; *address = (old >= val) ? 0 : old + 1; return old` |
| `unsigned atomicDec(unsigned* address, unsigned val)` | `u32` | `old = *address; *address = (old == 0 or old > val) ? val : old - 1; return old` |

### Bitwise（位运算）

| API | T | Func |
|---|---|---|
| `template<class T> T atomicAnd(T* address, T val)` | `i32, u32, u64` | `old = *address; *address &= val; return old` |
| `template<class T> T atomicOr(T* address, T val)` | `i32, u32, u64` | `old = *address; *address \|= val; return old` |
| `template<class T> T atomicXor(T* address, T val)` | `i32, u32, u64` | `old = *address; *address ^= val; return old` |

### CAS（Compare-And-Swap）

| API | T | Func |
|---|---|---|
| `template<class T> T atomicCAS(T* address, T compare, T val)` | `i32, u32, u64, u16, i32*, u64*` | `old = *address; if (old == compare) *address = val; return old` |

> `atomicCAS` 是其他所有原子操作的底层原语。通过它可以用 compare 条件自行实现任意原子逻辑。

### Tips

- **返回值为旧值**而非新值，注意不要搞反
- **System-Level**（`_system` 后缀，SM 6.x+）：所有原子函数都有 `_system` 版本（如 `atomicAdd_system`），保证跨 CPU-GPU 可见。性能比 device-scope 低，仅跨 CPU/GPU 通信时需要

---

## 编程实战

写一个 kernel 统计数组中正数的个数：

```cpp
__global__ void count_positive(const float *data, int *counter, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (data[idx] > 0.0f) {
        atomicAdd(counter, 1);
    }
}
```

简单单一场景，恰好展示 atomicAdd 最直接的用法——多个线程同时写同一个计数器，用原子操作保证不丢数据。

---

## 硬件实现（扩展）

### 原子指令（PTX）

在 PTX 层面，原子操作对应 `atom` 指令。`atomicAdd` 编译后是一条 `atom.global.add`：

```ptx
atom.global.add.u32 %r0, [%rd_addr], 1;  // [addr] += 1, %r0 = old([addr])
atom.global.add.f32 %f0, [%rd_addr], %f1; // [addr] += f1, %f0 = old([addr])
atom.global.add.s32 %r0, [%rd_addr], %r1; // [addr] += r1, %r0 = old([addr])
```

与非原子的对比：

```ptx
// 非原子：3 条指令，中间可被打断
ld.global.u32  %r1, [%rd_addr];  // 1. 读 ← 另一 core 可在此插入
add.u32        %r1, %r1, 1;      // 2. 改
st.global.u32  [%rd_addr], %r1;  // 3. 写

// 原子：1 条指令，不可打断
atom.global.add.u32 %r0, [%rd_addr], 1;  // 读 + 改 + 写 一气呵成
```

两个 core 同时执行原子指令，不会出现 race：

```ptx
atom.global.add.u32 %r0, [%rd_addr], 1;  // Core 1  [addr] = 11, %r0 = 10
atom.global.add.u32 %r0, [%rd_addr], 1;  // Core 2  [addr] = 12, %r0 = 11  ✓
```

### 不同 Scope 的原子

PTX 通过 scope 修饰符控制原子性的可见范围：

| PTX 指令 | 锁定范围 | 性能 |
|---|---|---|
| `atom.global.add.s32` | L2 Cache（全局可见） | 较慢 |
| `atom.shared.add.s32` | Shared Memory bank（仅 block 内可见） | 快 |
| `atom.global.add.s32.sys` | 系统总线（跨 CPU-GPU） | 最慢 |

### 硬件锁定机制

GPU 通过内存控制器锁定保证原子性：

| 方案 | 原理 | 适用场景 |
|---|---|---|
| **Lock Cache Line** | 锁定 L1/L2 缓存行，排他访问 | Global memory atomic，主流方案 |
| **Lock SRAM Bank** | 锁定 Shared memory bank | Shared memory atomic |
| **Lock 总线** | 原子期间锁定内存总线 | 较旧的 SM 架构 |

> 原子操作慢的原因就在这里——硬件需要做锁定和仲裁，同一时刻只有一个 atomic 能通过 memory controller。
