参考学习资料：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/programug/Ascendcopdevg/docs/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/%E4%BC%98%E7%A7%80%E5%AE%9E%E8%B7%B5/Matmul%E6%80%A7%E8%83%BD%E8%B0%83%E4%BC%98%E6%A1%88%E4%BE%8B/Matmul%E9%AB%98%E9%98%B6API%E4%BD%BF%E8%83%BDUnitFlag.md

使能UnitFlag的适用场景

算子的MMAD流水和FIXPIPE流水之间串行执行，FIXPIPE等待MMAD计算完成才搬出结果，这个指令同步等待的时间在算子整体执行耗时中占比较高。这种场景可以使能UnitFlag功能，以获得MMAD和FIXPIPE流水并行的性能收益。如果算子原本的MMAD、FIXPIPE流水可以被其他流水掩盖（比如MTE2 Bound），这时使能UnitFlag功能总体收益很小

设计优化方案
如下图所示，未开启UnitFlag功能时，MMAD和FIXPIPE是指令级别的同步，FIXPIPE指令需要等MMAD指令执行完成才进行结果搬出，MMAD和FIXPIPE之间流水串行。

图 1 未开启UnitFlag功能

![](https://raw.gitcode.com/cann/asc-devkit/raw/9.1.0-beta.1/docs/guide/figures/%E6%9C%AA%E5%BC%80%E5%90%AFUnitFlag%E5%8A%9F%E8%83%BD.png)

如下图所示，开启UnitFlag功能时，MMAD和FIXPIPE指令是512B大小的细粒度同步。在一条MMAD指令执行过程中，每当完成一个512B数据结果的计算，FIXPIPE立即开始搬出该512B的数据，从而实现MMAD和FIXPIPE之间的流水并行，提升算子性能。

图 2 开启UnitFlag功能

![](https://raw.gitcode.com/cann/asc-devkit/raw/9.1.0-beta.1/docs/guide/figures/%E5%BC%80%E5%90%AFUnitFlag%E5%8A%9F%E8%83%BD.png)
