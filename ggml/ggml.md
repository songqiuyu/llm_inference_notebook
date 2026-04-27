# 源码解读

## 核心数据结构

### ggml_object

```c
struct ggml_object {
    size_t offs;
    size_t size;

    struct ggml_object * next;

    enum ggml_object_type type;

    char padding[4];
};
```

`offs`：这个object的**数据**在`mem_buffer`中的字节偏移量，为什么会有32的偏移，object这个结构体相当于一个头，这个头是占了32个字节。
`size`：这个object占用的字节数
`next`：单向链表，指向的下一个object
`type`：枚举类型，主要有三种`TENSOR`、`GRAPH`和`WORK_BUFFER`

这里的`ggml_object`本身也存在ctx的`mem_buffer`里

### ggml_context

ggml_context是一个上下文管理结构体

```c
struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    int    n_objects;

    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;
};
```
ggml_context本质是一个线性内存池的管理器，这里面称为`arena allocator`，是一个控制头。

`mem_size`：是表示内存池的总大小（多少个字节），表示的是`mem_buffer`这片大内存的总容量上限

`mem_buffer`：是这块大内存的起始地址，一大块连续大内存，这块内存存放的是：`ggml_object`、`ggml_tensor`等等

`mem_buffer_owned`：这块内存是不是该ctx拥有的，如果说我声明一个ctx，没有给他分配内存，是ctx自己malloc的，那就有拥有权，如果指定了一个地址，那就没有拥有权。**拥有权在于允不允许ggml来释放**

`no_alloc`：如果是false，表示新建张量的时候，tensor的元数据+数据本身都分配在mem_buffer中，如果是true，那就是只在mem_buffer中存ggml_tensor的结构体，数据部分不去分配，数据可能要分配到后端上（目前是这样）

`n_object`：整个ctx就是一个链表，这个就是链表节点数

`objects_begin`：链表头节点

`objects_end`：链表尾节点


#### 理解整个ctx

![ctx](image.png)

如果不进行alloc，那整个ctx保留的就是元数据，这种方式让实际数据的存储与后端剥离开，不会先分配到内存中

### ggml_tensor

```c
struct ggml_tensor {
    enum ggml_type type;

    struct ggml_backend_buffer * buffer;

    int64_t ne[GGML_MAX_DIMS]; // number of elements
    size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                // nb[0] = ggml_type_size(type)
                                // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                // nb[i] = nb[i-1] * ne[i-1]

    // compute data
    enum ggml_op op;

    // op params - allocated as int32_t for alignment
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

    int32_t flags;

    struct ggml_tensor * src[GGML_MAX_SRC];

    // source tensor and offset for views
    struct ggml_tensor * view_src;
    size_t               view_offs;

    void * data;

    char name[GGML_MAX_NAME];

    void * extra; // extra things e.g. for ggml-cuda.cu

    char padding[8];
};
```

`type`：指的是数据的元素类型，决定的是每个元素占用多少字节
`buffer`：tensor指向的后端存储区
`ne`：number of element，指的是形状（一般是列、行，第三维，第四维）
`nb`：number of Bytes，每个维度存多少个字节，这个是方便stride的，方便求地址用
`ggml_op`：就是算子类型
`op_params`：运算参数




### ggml_backend_buffer

这个就是把数据实际存储到了该后端上

```c
struct ggml_backend_buffer {
    struct ggml_backend_buffer_i  iface;
    ggml_backend_buffer_type_t    buft;
    void * context;
    size_t size;
    enum ggml_backend_buffer_usage usage;
};
```

`ggml_backend_buffer_i`：这里面保存的是虚函数表，是一组虚函数指针，去定义这块内存怎么读写，是为了适配不同后端（CPU/CUDA/Metal）填入不同的函数实现，上层的代码去统一调用

`buft`：记录自己是哪种类型的buffer，用来查询对齐要求、名称等元信息

`context`：后端私有数据

`size`：这块buffer的总字节数

`usage`：用途标记


#### ggml_backend_buffer_i

```c
struct ggml_backend_buffer_i {
    // 释放底层内存
    void         (*free_buffer)  (ggml_backend_buffer_t buffer);
    // 返回内存起始地址
    void *       (*get_base)     (ggml_backend_buffer_t buffer);
    // 将tensor注册到buffer
    enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    // 对tensor数据区清零
    void         (*memset_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
    // 把CPU数据写入buffer
    void         (*set_tensor)   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    // 从buffer读取数据返回cpu
    void         (*get_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
    // (optional) 2d data copies
    void         (*set_tensor_2d)(...);
    void         (*get_tensor_2d)(...);
    // (optional) tensor copy between backends
    bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);
    // 整块buffer清零
    void         (*clear)        (ggml_backend_buffer_t buffer, uint8_t value);
    // (optional) reset any internal state
    void         (*reset)        (ggml_backend_buffer_t buffer);
};
```

#### 进一步理解

![alt text](image-1.png)

### ggml_opt_dataset

ggml_opt_dataset是最新版里面，用来初始化数据集的。

```c
struct ggml_opt_dataset {
    struct ggml_context   * ctx    = nullptr;
    ggml_backend_buffer_t   buf    = nullptr;
    struct ggml_tensor    * data   = nullptr;
    struct ggml_tensor    * labels = nullptr;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_data    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};
```

`ggml_context`：内存池管理器
`ggml_backend_buffer_t`：后端存储
`ggml_tensor`：数据tensor
`ggml_tensor`：标签tensor




### ggml_tensor



## mnist-eval.cpp

```cpp
    ggml_opt_dataset_t result = new ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = ggml_init(params);
    }

```

是去申请ggml_context，这个context你可以理解为一个ggml_obj的一个内存池，以链表的形式去存一个个的ggml_obj的对象

object有三种：tensor、graph和buffer

前面是把ggml_init给init好了

ggml_new_tensor_2d，是去申请context里面的data，这个data就是提前开辟好的内存

get_tensor的过程中`ggml_new_tensor_impl`，是返回ggml_tensor，主要做的工作是new一个ggml的object，可以看一下这个object，`static struct ggml_object * ggml_new_object(struct ggml_context * ctx, enum ggml_object_type type, size_t size)`

