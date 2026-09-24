// ============================================================================
// 语法检查用 stub —— 只为让本机能跑 g++ -fsyntax-only 验证 kernel.asc 的
// **自身语法与我方代码的内部一致性**（错字、漏分号、实参个数/类型、重载选择、
// 结构体字段名、作用域），不模拟 AscendC 的运行时语义。
//
// 关键：API 的**签名与字段名**按本地 CANN 8.5 文档抄写，这样"字段名写错/重载挑错"
// 这类错误能被 g++ 抓出来。不能证明真机能编译（真机还要看 NPU_ARCH、模板实参、
// 工具链特性），但能把"低级的编译错误"在提交前清掉。
// ============================================================================
#ifndef ASCEND_STUB_H
#define ASCEND_STUB_H

#include <cstdint>
#include <cstddef>

// ---- 语言扩展关键字 ----
#define __global__
#define __aicore__
#define __kfc_workspace__
#ifndef __gm__
#define __gm__
#endif
// 真机： #define ASCEND_IS_AIV constexpr(g_coreType == AscendC::AIV)
// （sys_macros.h:79）——它是**不可取反**的表达式；用括号形式近似，
// 并靠 source_check 的"禁止 !ASCEND_IS_*"断言把真机约束固化下来。
#define ASCEND_IS_AIV (g_coreType == AscendC::AIV)
#define ASCEND_IS_AIC (g_coreType == AscendC::AIC)
namespace AscendC { constexpr int AIV = 0; constexpr int AIC = 1;
                    static const int g_coreType = AIV; }

typedef uint8_t *GM_ADDR;
typedef void *aclrtStream;

namespace AscendC {

enum TPosition { VECIN, VECCALC, VECOUT, GM, CO1, CO2, MS, LCM, TSCM };
enum class CubeFormat { ND, NZ, ND_ALIGN };
enum class ReduceOrder { ORDER_VALUE_INDEX, ORDER_INDEX_VALUE, ORDER_ONLY_VALUE, ORDER_ONLY_INDEX };
enum PIPE { PIPE_ALL, PIPE_MTE1, PIPE_MTE2, PIPE_MTE3, PIPE_V, PIPE_S, PIPE_FIX };

enum class HardEvent : uint8_t {
    MTE2_MTE1, MTE1_MTE2, MTE1_M, M_MTE1, MTE2_V, V_MTE2, MTE3_V, V_MTE3,
    M_V, V_M, V_V, MTE3_MTE1, MTE1_MTE3, MTE1_V, MTE2_M, M_MTE2,
    MTE2_S, S_MTE2, MTE3_S, S_MTE3, V_S, S_V, FIX_V, V_FIX,
};

typedef uint8_t event_t;

struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen;
    uint32_t srcStride;
    uint32_t dstStride;
    uint32_t rsv;
};

// 真机字段名与 8.5 文档不一致（实测 rightPadValue/padValue 不存在）→ 本 stub 故意
// 只保留 isPad（其余字段名未知），这样"设字段"的代码在本地就会编译失败，
// 强制走"默认构造"这条真机可行路径。
template <typename T>
struct DataCopyPadExtParams {
    bool isPad;
};

template <typename T>
struct LocalTensor {
    T *p;
    LocalTensor<T> operator[](uint32_t) const;   // 真机返回切片（LocalTensor）
    T GetValue(uint32_t) const;
    void SetValue(uint32_t, T) const;
};

template <typename T>
struct GlobalTensor {
    void SetGlobalBuffer(__gm__ T *, uint64_t);
    GlobalTensor<T> operator[](uint64_t) const;
    T GetValue(uint64_t) const;
};

template <TPosition pos>
struct TBuf {
    template <typename T>
    LocalTensor<T> Get() const;
};

struct TPipe {
    void InitBuffer(TBuf<TPosition::VECCALC> &, uint32_t);
    uint32_t FetchEventID(HardEvent);
};

template <HardEvent E>
void SetFlag(event_t);
template <HardEvent E>
void WaitFlag(event_t);
template <PIPE P>
void PipeBarrier();

// ---- 设备侧基础 API（签名按 8.5 文档）----
template <typename T>
void Duplicate(const LocalTensor<T> &, const T &, int32_t);
template <typename T>
void ReduceMax(const LocalTensor<T> &, const LocalTensor<T> &, const LocalTensor<T> &,
               const int32_t, bool calIndex = false);

template <typename T>
void Max(const LocalTensor<T> &, const LocalTensor<T> &, const LocalTensor<T> &, int32_t);
template <typename T>
void WholeReduceMax(const LocalTensor<T> &, const LocalTensor<T> &, const int32_t mask,
                    const int32_t repeatTimes, const int32_t dstRepStride,
                    const int32_t srcBlkStride, const int32_t srcRepStride,
                    ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);
template <typename T>
void DataCopyPad(const LocalTensor<T> &, const GlobalTensor<T> &,
                 const DataCopyExtParams &, const DataCopyPadExtParams<T> &);
template <typename T>
void DataCopyPad(const GlobalTensor<T> &, const LocalTensor<T> &,
                 const DataCopyExtParams &);

void SetAtomicAdd();
template <typename T>
void SetAtomicAdd();
void SetAtomicNone();

uint32_t GetBlockIdx();
uint32_t GetBlockNum();
GM_ADDR GetSysWorkSpacePtr();

struct half {
    uint16_t v;
    operator float() const { return 0.0f; }
};
struct bfloat16_t { uint16_t v; };
float ToFloat(bfloat16_t);

// ---- Matmul 高阶 API ----
template <TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
          int LAYOUT = 0, bool IBSHARE = false>
struct MatmulType {
    using T = TYPE;
};

template <typename AType, typename BType, typename CType>
struct Matmul {
    void SetOrgShape(int32_t, int32_t, int32_t, int32_t = 0, int32_t = 0);
    void SetSingleShape(int32_t, int32_t, int32_t);
    void SetTensorA(const GlobalTensor<typename AType::T> &, bool);
    void SetTensorB(const GlobalTensor<typename BType::T> &, bool);
    bool Iterate();
    void GetTensorC(const GlobalTensor<float> &, uint8_t, bool);
    void IterateAll(const GlobalTensor<float> &, uint8_t = 0, bool = false,
                    bool = false, bool = false);
    void End();
};

namespace tiling {
struct TCubeTiling {
    int32_t usedCoreNum;
    int32_t M, N, Ka, Kb;
    int32_t singleCoreM, singleCoreN, singleCoreK;
    int32_t baseM, baseN, baseK;
    int32_t depthA1, depthB1;
    int32_t stepM, stepN, stepKa, stepKb;
    int32_t isBias;
    int32_t transLength;
    int32_t iterateOrder;
    int32_t dbL0A, dbL0B, dbL0C;
    int32_t shareMode, shareL1Size, shareL0CSize, shareUbSize;
    int32_t batchM, batchN, singleBatchM, singleBatchN;
};
} // namespace tiling

} // namespace AscendC

// ---- Kernel 类型宏（AIV/AIC 配比）----
enum KernelType {
    KERNEL_TYPE_AIV_ONLY, KERNEL_TYPE_AIC_ONLY,
    KERNEL_TYPE_MIX_AIC_1_2, KERNEL_TYPE_MIX_AIC_1_1,
    KERNEL_TYPE_MIX_AIC_1_0, KERNEL_TYPE_MIX_AIV_1_0,
};
#define KERNEL_TASK_TYPE_DEFAULT(x) do { (void)(x); } while (0)
#define KERNEL_TASK_TYPE(key, value) do { (void)(key); (void)(value); } while (0)

// REGIST_MATMUL_OBJ 宏：真机展开为框架注册，这里只需吃掉参数
#define REGIST_MATMUL_OBJ(pipe, ws, mm, tiling) do { (void)(pipe); (void)(ws); (void)(mm); (void)(tiling); } while (0)

// ---- matmul_tiling（host 侧）----
namespace platform_ascendc { class PlatformAscendC; }
namespace matmul_tiling {
enum class DataType { DT_FLOAT16, DT_BFLOAT16, DT_FLOAT };
enum class CubeFormat { ND, NZ };
enum class TPosition { GM, VECIN };
class MatmulApiTiling {
public:
    explicit MatmulApiTiling(platform_ascendc::PlatformAscendC &platform);
    int64_t SetAType(TPosition, CubeFormat, DataType, bool = false);
    int64_t SetBType(TPosition, CubeFormat, DataType, bool = false);
    int64_t SetCType(TPosition, CubeFormat, DataType);
    int64_t SetBias(bool);
    int64_t SetOrgShape(int32_t, int32_t, int32_t);
    int64_t SetShape(int32_t, int32_t, int32_t);
    int64_t SetFixSplit(int32_t, int32_t, int32_t);
    int64_t SetBufferSpace(int32_t, int32_t, int32_t, int32_t = -1);
    int64_t GetTiling(AscendC::tiling::TCubeTiling &);
};
} // namespace matmul_tiling

namespace platform_ascendc {
class PlatformAscendC {
public:
    uint32_t GetCoreNumAic() const;
    uint32_t GetCoreNumAiv() const;
    uint64_t GetLibApiWorkSpaceSize() const;
};
class PlatformAscendCManager {
public:
    static PlatformAscendC *GetInstance();
};
} // namespace platform_ascendc

// ---- ACL（host 侧）----
typedef int aclError;
#define ACL_SUCCESS 0
#define ACL_MEM_MALLOC_HUGE_FIRST 0
aclError aclrtMalloc(void **, size_t, int);
aclError aclrtMemset(void *, size_t, int, size_t);
aclError aclrtFree(void *);
aclError aclrtSynchronizeStreamWithTimeout(aclrtStream, int);
#define ACL_MEMCPY_DEVICE_TO_DEVICE 3
aclError aclrtMemcpy(void *, size_t, const void *, size_t, int);

#endif // ASCEND_STUB_H
