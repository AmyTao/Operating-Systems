#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

struct VirtualMemory {
  uchar *buffer;
  uchar *storage;
  u32 *invert_page_table;
  int *pagefault_num_ptr;
  u32 *swap_table;

  int PAGESIZE;
  int INVERT_PAGE_TABLE_SIZE;
  int PHYSICAL_MEM_SIZE;
  int STORAGE_SIZE;
  int PAGE_ENTRIES;
  int tid;
};

// TODO
__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES, int tid=0);
__device__ uchar vm_read(VirtualMemory *vm, u32 addr);
__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value);
__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size);

__device__ void swap_out(VirtualMemory *vm, u32 dst, u32 empty);
__device__ void swap_in(VirtualMemory *vm, u32 dst,u32 addr,int option);
__device__ bool is_page_in_memory(VirtualMemory *vm,u32 addr,int *pagefault_num_ptr,u32 *index);
__device__ bool is_page_in_disk(VirtualMemory *vm,u32 addr,int *pagefault_num_ptr,u32 *index,u32 *empty);
__device__ bool is_memory_full(VirtualMemory *vm,u32 *index);
__device__ void LRU_update(VirtualMemory *vm, int option,u32 addr);
__device__ bool have_empty_place(VirtualMemory *vm,u32 *empty);
__device__ bool is_disk_full(VirtualMemory *vm, u32 empty);
__device__ void init_invert_page_table(VirtualMemory *vm);









                        

#endif
