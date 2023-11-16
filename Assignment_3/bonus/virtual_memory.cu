#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


__device__ void init_invert_page_table(VirtualMemory *vm) {//pid =0 in task 1

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1, set dirty bit as 0
    //printf("%u\n",vm->invert_page_table[i]);
  }
  for(int j=vm->PAGE_ENTRIES;j<2*vm->PAGE_ENTRIES;j++){// LRU queue, reflect the time info about the memory
    vm->invert_page_table[j]=0x00200400; //---10---1---10---1---10
                                            //0000 0000 0010 0000 0000 0100 0000 0000
    //printf("%u\n",vm->invert_page_table[j]);

  }
  for(int k=2*vm->PAGE_ENTRIES;k<4*vm->PAGE_ENTRIES;k++){// pages in the VM (physical MEM+ disk part)
    vm->invert_page_table[k]=0x80008000;
    //printf("%u\n",vm->invert_page_table[k]);
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES,int tid) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;
  vm->tid=tid;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ void swap_out(VirtualMemory *vm, u32 dst, u32 empty){
  // swap out
  // option=0 means previously stored in disk and is read from disk
  // 1) store the data from memory at dst to disk at empty (all char in the page)
  // 2) update swap table(valid the empty)
  // 3) invalid the meomry space
  // empty= emptyline range from 2*page_entries-4*page*entries
  // dst is the page in memory
  u32 page=vm->invert_page_table[dst]&0x00007FFF; //0 0 0 0 0111 1111 1111 1111 15bit
  u32 option=(vm->invert_page_table[dst]&0x40000000)>>30; //0100 
  if(option==1){
    if((vm->invert_page_table[empty]&0x80000000)!=0){// if the first place is empty
      for(int i=0;i<32;i++){// one page is 32B, read byte by byte
        vm->storage[((2*(empty-2*vm->PAGE_ENTRIES))<<5)+i]=vm->buffer[(dst<<5)+i];
      }
      // update the swap table
      vm->invert_page_table[empty]=vm->invert_page_table[empty]&0x0000FFFF; //0 0 0 0 1 1 1 1  
      vm->invert_page_table[empty]=vm->invert_page_table[empty]|(page<<16); // update the VPN
    }else{// the second element in this entry 
      for(int i=0;i<32;i++){// one page is 32B
        vm->storage[((2*(empty-2*vm->PAGE_ENTRIES)+1)<<5)+i]=vm->buffer[(dst<<5)+i];
      }
      // update the swap table
      vm->invert_page_table[empty]=vm->invert_page_table[empty]&0xFFFF0000; //1111 1111 1111 1111 0 0 0 0
      vm->invert_page_table[empty]=vm->invert_page_table[empty]|(page);  // update the VPN
    }
  }
  // invalid the page dst in memory
  vm->invert_page_table[dst]=vm->invert_page_table[dst]|0x80000000;
}
__device__ void swap_in(VirtualMemory *vm, u32 dst,u32 addr,int option){ 
  // swap in
  //option=0 under read, 1 under write
  // 1) store data from the disk at addr to the memory at dst
  // 2) update the swap table (invalid the addr)
  // 3) update the page table,set dirty bit
  // addr=Physical page addr in disk
  // dst=PN in memory
  // change the memory

  u32 page;
  for(int i=0;i<32;i++){// one page is 32B, read byte by byte
    vm->buffer[(dst<<5)+i]=vm->storage[(addr<<5)+i];
  }

  // update the swap table
  if(addr%2==0){
    //get the VPN of the swap page
    page=(vm->invert_page_table[addr/2+2*vm->PAGE_ENTRIES]&0x7FFF0000)>>16; //0111 1111 1111 1111 first 15bit 
    //invalid the swap page in disk
    if(option==1){
    vm->invert_page_table[addr/2+2*vm->PAGE_ENTRIES]=vm->invert_page_table[addr/2+2*vm->PAGE_ENTRIES]|0x80000000; //1000 
    }
  }else{
    page=vm->invert_page_table[(addr-1)/2+2*vm->PAGE_ENTRIES]&0x00007FFF; //0111 1111 1111 1111 last 15bit
    if(option==1){
    vm->invert_page_table[(addr-1)/2+2*vm->PAGE_ENTRIES]=vm->invert_page_table[(addr-1)/2+2*vm->PAGE_ENTRIES]|0x00008000; //0 0 0 0 1000 
    }
  }
  //update the page table, set VPN as the swap page
  vm->invert_page_table[dst]=vm->invert_page_table[dst]&0x3FFF0000; //0011 1111 1111 1111 0 0 0 0 set 0
  vm->invert_page_table[dst]=vm->invert_page_table[dst]|page;
  if(option==1){// if write, set the dirty bit as 1
  vm->invert_page_table[dst]=vm->invert_page_table[dst]|0x40000000;     //0100 
  }
}

__device__ bool is_page_in_memory(VirtualMemory *vm,u32 addr,int *pagefault_num_ptr,u32 *index){
  //check from head to tail
    u32 head=(vm->invert_page_table[vm->PAGE_ENTRIES]&0xFFC00000)>>22; //0|    1111 1111 1100 0 0 0 0 0
    u32 tail=(vm->invert_page_table[vm->PAGE_ENTRIES+1]&0xFFC00000)>>22;// 1|  1111 1111 1100 0 0 0 0 0
    u32 ptr=head;//get the head of the LRU list
    u32 page;
    // if tail==ptr
    // 1) head==tail (empty, 1)
    // 2) end
    //start from head
    //page=ptr;
    //printf("%u\n",addr);
    while(tail!=ptr){// more than 2 pages in memory
      //printf("here3\n");
      if((vm->invert_page_table[ptr]&0x00007FFF)==addr){ //0000 0000 0000 0000 0111 1111 1111 1111 15 bits
      // if the VPN equals the dst
        *index=ptr;// return the index in memory range from 0-2**10-1
        return true;
      }
      // next page 
      ptr=vm->invert_page_table[vm->PAGE_ENTRIES+ptr]&0x000003FF; // 0 0 0 0 0 0011 1111 1111
      //printf("%u\n",ptr);
      //page=(vm->invert_page_table[vm->PAGE_ENTRIES+ptr]&0x001FF800)>>11;// 0000 0000 0001 1111 1111 1000 0000 0000
      }
      // if the queue is empty
      if(((vm->invert_page_table[vm->PAGE_ENTRIES+head]&0x00200000)>>21)==1){ //0000 0000 0010 0 0 0 0 0
        //printf("here1\n");
        (*pagefault_num_ptr)++;
        return false;
      }
      // if the queue has 1 element
      if(((vm->invert_page_table[vm->PAGE_ENTRIES+head]&0x00000400)>>10)==1){ //0000 0000 0000 0000 0000 0100 0000 0000
        page=(vm->invert_page_table[vm->PAGE_ENTRIES+head]&0x001FF800)>>11;//0000 0000 0001 1111 1111 1000 0 0 
        //printf("%u\n",page);
        //printf("%u\n",vm->invert_page_table[page]&0x00007FFF);
        //printf("%u\n",addr);
        if((vm->invert_page_table[page]&0x00007FFF)==addr){ //0000 0000 0000 0000 0111 1111 1111 1111 15 bits
        // if the VPN equals the dst
          *index=page;// return the index in memory range from 0-2**10-1
          return true;
        }
        else{
          (*pagefault_num_ptr)++;
          //printf("here2\n");
          return false;
        }
      }
      // if the tail is not the target
      //printf("here4\n");
      if((vm->invert_page_table[ptr]&0x00007FFF)==addr){ //0000 0000 0000 0000 0111 1111 1111 1111 15 bits
      // if the VPN equals the dst
        *index=page;// return the index in memory range from 0-2**10-1
        return true;
      }else{
        (*pagefault_num_ptr)++;
        return false;
      }
}

__device__ bool is_page_in_disk(VirtualMemory *vm,u32 addr,int *pagefault_num_ptr,u32 *index,u32 *empty){ 
  // addr=target VPN
  // returns: index=hit addr in disk, empty=empty line in swap table
  *empty=4*vm->PAGE_ENTRIES;
  bool flag1=false;// for finding the page
  bool flag2=false;// for finding the empty space
  for(int i=2*vm->PAGE_ENTRIES;i<4*vm->PAGE_ENTRIES;i++){
    if(flag1&&flag2) break;
    if((vm->invert_page_table[i]&0x80000000)==0){// the first element in this entry
      if(((vm->invert_page_table[i]&0x7FFF0000)>>16)==addr){// if the VPN is the same 0111 1111 1111 1111 0 0 0 0        
        *index=(i-2*vm->PAGE_ENTRIES)*2;// represent hit addr
        flag1=true;
      }
    }
    else{
      if(i<(*empty)){
        *empty=i;//represent empty line
        flag2=true;
      }
    }
    if((vm->invert_page_table[i]&0x00008000)==0){// the second element in this entry 
      if((vm->invert_page_table[i]&0x00007FFF)==addr){// if the VPN is the same 0 0 0 0 0111 1111 1111 1111
        *index=(i-2*vm->PAGE_ENTRIES)*2+1;// represent hit addr
        flag1=true;
      }
    }
    else{
      if(i<(*empty)){
        *empty=i;//represent empty pointer
        flag2=true;
      }
    }
  }
  return flag1;
}
__device__ bool is_memory_full(VirtualMemory *vm,u32 *index){ // index is the empty pointer in memory
  if((vm->invert_page_table[2+vm->PAGE_ENTRIES]&0xFFC00000)>>22==0x000003FF){ // check the count num 1111 1111 1100 0 0 0 0 0 whether it equals 0 0 0 0 0 0011 1111 1111
    return true;
  }else{
    u32 i=0;
    //printf("%u\n",vm->invert_page_table[i]&0x80000000);
    while(((vm->invert_page_table[i]&0x80000000)==0)&&(i<vm->PAGE_ENTRIES)){//check the valid bit to find a empty space
      //printf("itier\n");
      i++;
    }
    *index=i;// index point to the empty space 
    return false;
  }
}
__device__ bool is_disk_full(VirtualMemory *vm, u32 empty){
  if(empty==4*vm->PAGE_ENTRIES) return true;
  else{return false;}
}
__device__ bool have_empty_place(VirtualMemory *vm,u32 *empty){
  for(int i=0;i<vm->PAGE_ENTRIES;i++){
    if(((vm->invert_page_table[i]&0x80000000)==0x80000000)||((vm->invert_page_table[i]&0x40000000)==0x0)){
      //0100
      *empty=i;
      return true;
  }
  } 
  printf("%u\n",vm->invert_page_table[vm->PAGE_ENTRIES-1]&0x40000000);
  return false;

}
__device__ void LRU_update(VirtualMemory *vm, int option,u32 addr){
  //addr is the index in page table 0-2*10-1
    u32 tail=(vm->invert_page_table[vm->PAGE_ENTRIES+1]&0xFFC00000)>>22; // 1111 1111 1100 0 0 0 0 0
    u32 head=(vm->invert_page_table[vm->PAGE_ENTRIES]&0xFFC00000)>>22; // 1111 1111 1100 0 0 0 0 0
    u32 count=(vm->invert_page_table[vm->PAGE_ENTRIES+2]&0xFFC00000)>>22;
    u32 page;
    u32 prev;
  if(option==0){ // case 1: target page is in the memory, at the head of the LRU queue. >> do nothing
    return;
  }
  if(option==1){ // case 2: target page is in the memory, at the tail of the LRU queue.
    //get the tail.prev (at least we have 2 nodes)
    page=(vm->invert_page_table[tail+vm->PAGE_ENTRIES]&0x001FF800)>>11; //0000 0000 0001 1111 1111 1000 0000 0000
    // invalid the tail.prev.next
    vm->invert_page_table[page+vm->PAGE_ENTRIES]=vm->invert_page_table[page+vm->PAGE_ENTRIES]|0x00000400;  //0 0 0 0 0 0100 0 0 set 1
    //tail.next=head
    vm->invert_page_table[tail+vm->PAGE_ENTRIES]=vm->invert_page_table[tail+vm->PAGE_ENTRIES]&0xFFFFF800; //1111 1111 1111 1111 1111 1000 0 0 set 0
    vm->invert_page_table[tail+vm->PAGE_ENTRIES]=vm->invert_page_table[tail+vm->PAGE_ENTRIES]|head; //update the tail node
    //valid newhead(tail)
    vm->invert_page_table[tail+vm->PAGE_ENTRIES]=vm->invert_page_table[tail+vm->PAGE_ENTRIES]&0xFFDFFFFF; //1 1 1101 1 1 1 1 1
    //tail=tail.prev
    vm->invert_page_table[1+vm->PAGE_ENTRIES]=vm->invert_page_table[1+vm->PAGE_ENTRIES]&0x003FFFFF; //0000 0000 0011
    vm->invert_page_table[1+vm->PAGE_ENTRIES]=vm->invert_page_table[1+vm->PAGE_ENTRIES]|(page<<22); //0000 0000 0011
    //head.prev=tail
    vm->invert_page_table[head+vm->PAGE_ENTRIES]=vm->invert_page_table[head+vm->PAGE_ENTRIES]&0xFFC007FF; //1111 1111 1100 0000 0000 0111 1111 1111 set 0
    vm->invert_page_table[head+vm->PAGE_ENTRIES]=vm->invert_page_table[head+vm->PAGE_ENTRIES]|(tail<<11); //update the tail node
    //head=tail
    vm->invert_page_table[vm->PAGE_ENTRIES]=vm->invert_page_table[vm->PAGE_ENTRIES]&0x003FFFFF; //0000 0000 0011 1111 1111 1111 1111 1111 set zero
    vm->invert_page_table[vm->PAGE_ENTRIES]=vm->invert_page_table[vm->PAGE_ENTRIES]|(tail<<22);  // update the head
  }
  if(option==2){ // case 3: target page is in the memory, in the middle of the LRU queue
  //get the page=addr.next
  page=vm->invert_page_table[addr+vm->PAGE_ENTRIES]&0x000003FF; //0000 0000 0000 0000 0000 0011 1111 1111
  //addr.prev.next=addr.next
  prev=(vm->invert_page_table[addr+vm->PAGE_ENTRIES]&0x001FF800)>>11; //0000 0000 0001 1111 1111 1000 0000 0000
  vm->invert_page_table[prev+vm->PAGE_ENTRIES]=vm->invert_page_table[prev+vm->PAGE_ENTRIES]&0xFFFFF800;//1111 1111 1111 1111 1111 1000 0 0 set 0
  vm->invert_page_table[prev+vm->PAGE_ENTRIES]=vm->invert_page_table[prev+vm->PAGE_ENTRIES]|page; //update the addr.prev
  //addr.next.prev=addr.prev
  vm->invert_page_table[page+vm->PAGE_ENTRIES]=vm->invert_page_table[page+vm->PAGE_ENTRIES]&0xFFC007FF; //1111 1111 1100 0000 0000 0111 1111 1111 set 0
  vm->invert_page_table[page+vm->PAGE_ENTRIES]=vm->invert_page_table[page+vm->PAGE_ENTRIES]|(prev<<11); //update the addr.next
  //addr.next=head
  vm->invert_page_table[addr+vm->PAGE_ENTRIES]=vm->invert_page_table[addr+vm->PAGE_ENTRIES]&0xFFFFF800;//1111 1111 1111 1111 1111 1000 0 0 set 0
  vm->invert_page_table[addr+vm->PAGE_ENTRIES]=vm->invert_page_table[addr+vm->PAGE_ENTRIES]|head; //update the head
  //head.prev=addr
  vm->invert_page_table[head+vm->PAGE_ENTRIES]=vm->invert_page_table[head+vm->PAGE_ENTRIES]&0xFFC007FF; //1111 1111 1100 0000 0000 0111 1111 1111 set 0
  vm->invert_page_table[head+vm->PAGE_ENTRIES]=vm->invert_page_table[head+vm->PAGE_ENTRIES]|(addr<<11); //update the head.prev
  //valid addr
  vm->invert_page_table[addr+vm->PAGE_ENTRIES]=vm->invert_page_table[addr+vm->PAGE_ENTRIES]&0xFFDFFFFF; //1 1 1101 1 1 1 1 1
  //head=addr
  vm->invert_page_table[vm->PAGE_ENTRIES]=vm->invert_page_table[vm->PAGE_ENTRIES]&0x003FFFFF; //0000 0000 0011 1111 1111 1111 1111 1111 set zero
  vm->invert_page_table[vm->PAGE_ENTRIES]=vm->invert_page_table[vm->PAGE_ENTRIES]|(addr<<22);  // update the head
  }
  if(option==3){ // case 4: target page is in disk or not in disk, memory is not full addr=empty space
    if(((vm->invert_page_table[head+vm->PAGE_ENTRIES]&0x00200000)>>21)==0){ //0000 0000 0010 0 0 0 0 0 if head is valid
        //printf("addr:%u\n",addr);//should be 1
        vm->invert_page_table[head+vm->PAGE_ENTRIES]=vm->invert_page_table[head+vm->PAGE_ENTRIES]&0xFFC007FF; //1111 1111 1100 0000 0000 0111 1111 1111 set 0
        vm->invert_page_table[head+vm->PAGE_ENTRIES]=vm->invert_page_table[head+vm->PAGE_ENTRIES]|(addr<<11); //update the head.prev
        // allocate a new node in LRU
        vm->invert_page_table[addr+vm->PAGE_ENTRIES]=vm->invert_page_table[addr+vm->PAGE_ENTRIES]&0xFFDFFFFF; //1111 1111 1101 1111 valid the head
        //valid the next node
        vm->invert_page_table[addr+vm->PAGE_ENTRIES]=vm->invert_page_table[addr+vm->PAGE_ENTRIES]&0xFFFFFBFF; //1111 1111 1111 1111 1111 1011 1111 1111       
        // addr.next=head
        vm->invert_page_table[addr+vm->PAGE_ENTRIES]=vm->invert_page_table[addr+vm->PAGE_ENTRIES]&0xFFFFF800; // 1 1 1 1 1 1000 0000 0000
        vm->invert_page_table[addr+vm->PAGE_ENTRIES]=vm->invert_page_table[addr+vm->PAGE_ENTRIES]|head; // 1 1 1 1 1 1000 0000 0000
        // update head
        vm->invert_page_table[vm->PAGE_ENTRIES]=vm->invert_page_table[vm->PAGE_ENTRIES]&0x003FFFFF;//0000 0000 0011 1111 1 1 1 1 set 0
        vm->invert_page_table[vm->PAGE_ENTRIES]=vm->invert_page_table[vm->PAGE_ENTRIES]|(addr<<22); // update the head
        //head.prev=addr
        vm->invert_page_table[head+vm->PAGE_ENTRIES]=vm->invert_page_table[head+vm->PAGE_ENTRIES]&0xFFC007FF; //1111 1111 1100 0000 0000 0111 1111 1111
        vm->invert_page_table[head+vm->PAGE_ENTRIES]=vm->invert_page_table[head+vm->PAGE_ENTRIES]|(addr<<11); //1111 1111 1100 0000 0000 0111 1111 1111


      if(count==0){// if only one node inside the queue, tail should be initiated
        //set tail as the old head
        vm->invert_page_table[1+vm->PAGE_ENTRIES]=vm->invert_page_table[1+vm->PAGE_ENTRIES]&0x003FFFFF; //0000 0000 0011
        vm->invert_page_table[1+vm->PAGE_ENTRIES]=vm->invert_page_table[1+vm->PAGE_ENTRIES]|(head<<22); //0000 0000 0011
      }
      //count++
      vm->invert_page_table[2+vm->PAGE_ENTRIES]=vm->invert_page_table[2+vm->PAGE_ENTRIES]+0x00400000; //0000 0000 0100 0 0 0 0 0
    }else{// if the list is empty
      //valid the head
      vm->invert_page_table[addr+vm->PAGE_ENTRIES]=vm->invert_page_table[addr+vm->PAGE_ENTRIES]&0xFFDFFFFF; //1111 1111 1101 1111 
      // update the head
      vm->invert_page_table[vm->PAGE_ENTRIES]=vm->invert_page_table[vm->PAGE_ENTRIES]&0x003FFFFF; //0000 0000 0011 1111 1111 1111 1111 1111 set zero
      vm->invert_page_table[vm->PAGE_ENTRIES]=vm->invert_page_table[vm->PAGE_ENTRIES]|(addr<<22);  // update the head
    }
  }
  if(option==4){// case 5: target page is in disk or not in disk (initilization), memory is full
    //get the tail.prev
    page=(vm->invert_page_table[tail+vm->PAGE_ENTRIES]&0x001FF800)>>11;
    // invalid the tail.prev.next
    vm->invert_page_table[page+vm->PAGE_ENTRIES]=vm->invert_page_table[page+vm->PAGE_ENTRIES]|0x00000400;  //0 0 0 0 0 0100 0 0
    // tail= tail.prev 0000 0000 0001 1111 1111 1000 0000 0000
    vm->invert_page_table[1+vm->PAGE_ENTRIES]=vm->invert_page_table[1+vm->PAGE_ENTRIES]&0x003FFFFF; //0000 0000 0011 set 0
    vm->invert_page_table[1+vm->PAGE_ENTRIES]=vm->invert_page_table[1+vm->PAGE_ENTRIES]|(page<<22); //0000 0000 0011
    // at the place of tail, write a new head
    //valid the head
    vm->invert_page_table[tail+vm->PAGE_ENTRIES]=vm->invert_page_table[tail+vm->PAGE_ENTRIES]&0xFFDFFFFF; //1111 1111 1101 1111 
     //set the head.next
    vm->invert_page_table[tail+vm->PAGE_ENTRIES]=vm->invert_page_table[tail+vm->PAGE_ENTRIES]&0xFFFFF800; // 1 1 1 1 1 1000 0000 0000
    vm->invert_page_table[tail+vm->PAGE_ENTRIES]=vm->invert_page_table[tail+vm->PAGE_ENTRIES]|head; // 1 1 1 1 1 1000 0000 0000
    // update the head
    vm->invert_page_table[vm->PAGE_ENTRIES]=vm->invert_page_table[vm->PAGE_ENTRIES]&0x003FFFFF; //0000 0000 0011 1111 1111 1111 1111 1111 set zero
    vm->invert_page_table[vm->PAGE_ENTRIES]=vm->invert_page_table[vm->PAGE_ENTRIES]|(tail<<22);  // update the head, addr=space in memory
    //set the head.prev=tail
    vm->invert_page_table[head+vm->PAGE_ENTRIES]=vm->invert_page_table[head+vm->PAGE_ENTRIES]&0xFFC007FF; //1111 1111 1100 0000 0000 0111 1111 1111
    vm->invert_page_table[head+vm->PAGE_ENTRIES]=vm->invert_page_table[head+vm->PAGE_ENTRIES]|(tail<<11); //1111 1111 1100 0000 0000 0111 1111 1111
   

  }
}
__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  //pid(2)--VPN(13)--offset(5)
  u32 VPN=(addr&0x0003FFE0)>>5; //0000 0000 0000 0011 1111 1111 1110 0000
  VPN=(VPN|(threadIdx.x<<13));
  u32 offset=addr&0x0000001F;//0000 0000 0000 0000 0000 0000 0001 1111
  u32 index;
  u32 PN;
  u32 empty;
  int value;
  u32 head=(vm->invert_page_table[vm->PAGE_ENTRIES]&0xFFC00000)>>22; //0|    1111 1111 1100 0 0 0 0 0
  u32 tail=(vm->invert_page_table[vm->PAGE_ENTRIES+1]&0xFFC00000)>>22;// 1|  1111 1111 1100 0 0 0 0 0
  u32 count=(vm->invert_page_table[vm->PAGE_ENTRIES+2]&0xFFC00000)>>22;
  u32 memaddr;
  int option=0;
  if(is_page_in_memory(vm,VPN,vm->pagefault_num_ptr,&index)){
    PN=(index<<5)+offset;
    //value=*((int *)(&(vm->buffer[PN])));
    memcpy(&value,&(vm->buffer[PN]),4);
    // decide use what way to update LRU
    if(head==index){
    LRU_update(vm,0,index);    
    }else if(tail==index&&count>=1){// more than 2 nodes
    LRU_update(vm,1,index);
    }else{

    LRU_update(vm,2,index);
     
    }
  }//page fault
  else if(is_page_in_disk(vm,VPN,vm->pagefault_num_ptr,&index,&empty)){// index=hit addr, empty=empty line in swap table
    PN=(index<<5)+offset;
    memcpy(&value,&(vm->storage[PN]),4);
    if(is_memory_full(vm,&memaddr)){
    // if dirty--
        // if disk not full --swapout
          // swap in at tail
        // if disk full 
        // if there is a not dirty place--swap in at here
        // else error
       
    // if not dirty-swap in at tail
    if(((vm->invert_page_table[tail]&0x40000000)>>30)==1){ //0100 //if dirty
    // if tail in disk-->write at its own place
    u32 meaningless;
    if(is_page_in_disk(vm,(vm->invert_page_table[tail]&0x00007FFF),vm->pagefault_num_ptr,&empty,&meaningless)){
      for(int i=0;i<32;i++){// one page is 32B, read byte by byte
        vm->storage[(empty<<5)+i]=vm->buffer[(tail<<5)+i];
      }
    }
    else{// tail not in disk
      if(is_disk_full(vm,empty)){
          if(have_empty_place(vm,&empty)){
            tail=empty;
            if(head==empty) option=0;// index at head
            else{option=2;}// index in middle           
          }else{
            printf("%u\n",addr);
            printf("Error1\n");
          }
      }else{
        swap_out(vm,tail,empty);// swap out the page to disk
        //swap_in(vm,tail,index,0); // swap in the page to memory
        option=4;

      }
    }      
    }
    //swap_out(vm,tail,empty);// swap out the page to disk
    swap_in(vm,tail,index,0); // swap in the page to memory
    LRU_update(vm,option,tail);
    //LRU_update(vm,4,index); //here index is meaningless, because we update the tail
    }else{
    swap_in(vm,memaddr,index,0); // to write in memory, we need dst in memory, VPN, and option
    LRU_update(vm,3,memaddr);     
    }
  }else{
    printf("ERROR\n");
  }
  return value; //TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  //get the page number
  //pid(2)--VPN(13)--offset(5)
    u32 VPN=(addr&0x0003FFE0)>>5; //0000 0000 0000 0011 1111 1111 1110 0000 
    VPN=(VPN|(threadIdx.x<<13));
    //printf("VPN");
    //printf("%u\n",VPN);
    u32 offset=addr&0x0000001F;//0000 0000 0000 0000 0000 0000 0001 1111
    u32 index;
    u32 PN;
    u32 empty;
    u32 head=(vm->invert_page_table[vm->PAGE_ENTRIES]&0xFFC00000)>>22; //0|    1111 1111 1100 0 0 0 0 0
    u32 tail=(vm->invert_page_table[vm->PAGE_ENTRIES+1]&0xFFC00000)>>22;// 1|  1111 1111 1100 0 0 0 0 0
    u32 count=(vm->invert_page_table[vm->PAGE_ENTRIES+2]&0xFFC00000)>>22;
    u32 memaddr;
    int option;
    if(is_page_in_memory(vm,VPN,vm->pagefault_num_ptr,&index)){// index is the hit addr

      //find the memory place
      //printf("page in memory\n");
      PN=(index<<5)+offset;
      //write in memory
      vm->buffer[PN]=value;
      if(head==index){// no need to update the LRU queue
        //printf("0\n");
        LRU_update(vm,0,index);    
      }else if(tail==index&&count>=1){// more than 2 nodes
        LRU_update(vm,1,index);
      }else{
        LRU_update(vm,2,index);     
      }
    }
    else if(is_page_in_disk(vm,VPN,vm->pagefault_num_ptr,&index,&empty)){// index is the hit addr, empty is the empty line
      //PN=(index<<5)+offset;
      // write in disk
      //vm->storage[PN]=value;
      if(is_memory_full(vm,&memaddr)){
    // if tail is dirty--
        // if disk not full --swapout
          // swap in at tail
        // if disk full 
        // if there is a not dirty place--swap in at here
        // else error
       
    // if not dirty-swap in at tail
    if(((vm->invert_page_table[tail]&0x40000000)>>30)==1){ //0100 //if dirty
    // if tail in disk-->write at its own place
    u32 meaningless;
    if(is_page_in_disk(vm,(vm->invert_page_table[tail]&0x00007FFF),vm->pagefault_num_ptr,&empty,&meaningless)){
      for(int i=0;i<32;i++){// one page is 32B, read byte by byte
        vm->storage[(empty<<5)+i]=vm->buffer[(tail<<5)+i];
      }
    }
    else{// tail not in disk
      if(is_disk_full(vm,empty)){
          if(have_empty_place(vm,&empty)){
            tail=empty;
            if(head==empty) option=0;// index at head
            else{option=2;}// index in middle           
          }else{
            printf("Error2\n");
          }
      }else{
        swap_out(vm,tail,empty);// swap out the page to disk
        //swap_in(vm,tail,index,0); // swap in the page to memory
        option=4;

      }
    }      
    }

    //swap_out(vm,tail,empty);// swap out the page to disk
    swap_in(vm,tail,index,0); // swap in the page to memory
    LRU_update(vm,option,tail);
    //LRU_update(vm,4,index); //here index is meaningless, because we update the tail

      PN=(tail<<5)+offset;
      //write in memory
      vm->buffer[PN]=value;

      //swap out a page to the disk
      //swap_out(vm,tail,empty);// swap out the page to disk
      //swap in the page to the memory
      //swap_in(vm,tail,index,1); // swap in the page to memory
      // LRU update
      //LRU_update(vm,4,index); //here index is meaningless, because we update the tail
      }else{
      //swap in the page to the memory
      swap_in(vm,memaddr,index,1); //index is the hit addr
      LRU_update(vm,3,memaddr); 
      PN=(memaddr<<5)+offset;
      //write in memory
      vm->buffer[PN]=value;
    
      }
    }else{// page is not in disk, at the begining
      if(is_memory_full(vm,&memaddr)){
    // if tail is dirty--
        // if disk not full --swapout
        // if disk full 
        // if there is a not dirty place--tail=empty lru update
        // else error
       
    // if not dirty--skip lru update at tail
    if(((vm->invert_page_table[tail]&0x40000000)>>30)==1){ //0100 //if dirty
    // if tail in disk-->write at its own place
    u32 meaningless;
    if(is_page_in_disk(vm,(vm->invert_page_table[tail]&0x00007FFF),vm->pagefault_num_ptr,&empty,&meaningless)){
      for(int i=0;i<32;i++){// one page is 32B, read byte by byte
        vm->storage[(empty<<5)+i]=vm->buffer[(tail<<5)+i];
      }
    }
    else{// tail not in disk
      if(is_disk_full(vm,empty)){
          if(have_empty_place(vm,&empty)){
            tail=empty;
            if(head==empty) option=0;// index at head
            else{option=2;}// index in middle           
          }else{
            printf("Error3\n");
          }
      }else{
        swap_out(vm,tail,empty);// swap out the page to disk
        //swap_in(vm,tail,index,0); // swap in the page to memory
        option=4;

      }
    }      
    }


        vm->buffer[(tail<<5)+offset]=value;
        //write in page table
        //valid the place set the PN, set the dirty bit as 1
        vm->invert_page_table[tail]=0x40000000; //0100 
        vm->invert_page_table[tail]=vm->invert_page_table[tail]|VPN;       
        //update the LRU queue
        //
        LRU_update(vm,option,tail);
        //LRU_update(vm,4,index);
      }else{
      //find a place in memory
      vm->buffer[memaddr<<5+offset]=value;
      ////valid the place, set the PN, set the dirty bit as 1
      vm->invert_page_table[memaddr]=0x40000000; //0100 
      vm->invert_page_table[memaddr]=vm->invert_page_table[memaddr]|VPN;
      //update the LRU queue
      LRU_update(vm,3,memaddr);   
      head=(vm->invert_page_table[vm->PAGE_ENTRIES]&0xFFC00000)>>22;
   
      }
    }
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
   for(int i=0;i<input_size;i++){
    results[i]=vm_read(vm,offset+i);
   }

}

