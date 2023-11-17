#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 empty_block_record = 0;
__device__ __managed__ uchar temp[128];




__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

// initialize the superblock
// 1--used
// 0--empty
  for(int i=0;i<SUPERBLOCK_SIZE;i++){
    fs->volume[i]=0;
  }
// initilize the FCB entries
// -------------------------------------------------------------------------------------------------------------------------
// 0: 0                    19|   20 21     | 22 23 24  |  25  26   |  27   28   |             29            |    30   31      |
//      filename              block address  file size   create #    modified #  file type/ # of file in dir  parent dir addr
// .......
// 1023: ......
// -------------------------------------------------------------------------------------------------------------------------
for (int i=0;i<FCB_ENTRIES;i++){
  // set file size to 0xffffff
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22] = 0xff; 
  // set the file type
	  fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 29] = 0xff; 
}
}

__device__ bool is_file_found(char *s1, uchar *s2){
  while((*s1!='\0')&&(*s2!='\0')&&(*s1==*s2)){
    s1++;
    s2++;
  }
  if((*s1=='\0')&&(*s2=='\0')) return true;
  else{return false;}
}


__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
  // initilaize the empty_pointer
  u32 empty_ptr=fs->FCB_ENTRIES;
  // get first FCB entries's filename
  uchar* fcb_name=fs->volume+(fs->SUPERBLOCK_SIZE);
  // temp to store *s
  char* temp=s;
  // filename counter
  int count_filename=0;
  //iterate gtime to find the FCB entries
  for(int i=0;i<fs->FCB_ENTRIES;i++){
  // check valid bit and file_type
  if((fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22]!=0xff)&&(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 29]==0xff)){
  // valid
    if(is_file_found(s, fcb_name)) {
      //printf("File exist\n");
      return i;// return the inumber if the file is found
    }
  }else{
    // if the filename doesn't match
    // update empty_pointer for only one time
    if((empty_ptr==fs->FCB_ENTRIES)&&(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22]==0xff)) empty_ptr=i; 
  }
  // update the file name
    fcb_name+=32;
  }
  // not found: check option
  if(op==G_WRITE){
  // check empty_ptr
  // if the option is G_write and there is no empty FCB entries left
  // then there is error
  if(empty_ptr==fs->FCB_ENTRIES){
    printf("ERROR! No empty ptr for the opening a new file\n");
    return empty_ptr;
  }  
  else{ 
    // find empty FCB
    // check the file number of the parent directory (bonus)
    // if the file # in the parent directory >= 50 --> error
    // if the file # in the parent directory < 50
    // valid a file
    //printf("fp: %u\n",empty_ptr);
    // reset the file size
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * empty_ptr + 22]=0;
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * empty_ptr + 23]=0;
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * empty_ptr + 24]=0;
    // set create time using gtime
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * empty_ptr + 25]=gtime/256;
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * empty_ptr + 26]=gtime%256;
    // set modified time as 0
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * empty_ptr + 27]=gtime/256;
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * empty_ptr + 28]=gtime%256;
    // set the gtime
    gtime++;

    //printf("gtime: %u\n",gtime);
    // set the filename
    while(*temp!='\0'){
    // write the filename
    //printf("temp: %c\n",*temp);
      fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * empty_ptr + count_filename]=*temp;
    //printf("stored: %c\n",(char)fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * empty_ptr + count_filename]);
      count_filename++;
      temp++;
      if(count_filename==19 && *temp!='\0'){
        printf("ERROR, the file name is too large! Please change it later\n");
    // Still set a file with incomplete filename
        return empty_ptr;
      }
    }
   // printf("FINISH\n");
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * empty_ptr + count_filename]='\0';
  // set the parent directory(bonus)
  // update parent directory's file size(bonus)
  // update parent directory's file number(bonus)
    return  empty_ptr;
  }
  }else{
    // if the option is G_read and the file cannot be found, then there is error
    printf("ERROR! The given file descripter is invalid for open(read)\n");
  }

}

__device__ u32 fs_allocate_blocks(FileSystem *fs, u32 new_size, u32 fp){
  // the file size is checked beforehand in write
  // condition for new_size
  // 0) old size is 0, just return the empty block pointer
  // 1) new_size=0 used in RM
  // 2) new_size==old_size  hooray! nothing changes
  // 3) new_size< old_size shift forward all the following blocks after the target block
  // 4) new_size> old_size shift backward all the following blocks after the target block
  u32 FCB_address=fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp;
  u32 FCB_file_size=(fs->volume[FCB_address+22]<<16)+(fs->volume[FCB_address+23]<<8)+fs->volume[FCB_address+24];
  if(FCB_file_size==0) return empty_block_record;// a new file will use the first empty block after all used blocks
  else{
    // get the first block
    u32 block_address=(fs->volume[FCB_address+20]<<8)+fs->volume[FCB_address+21];
    // get total number of blocks
    u32 old_block_used=FCB_file_size/32;
    if(FCB_file_size%32!=0) old_block_used++;
    // block_end is right after the last block of the current file
    u32 block_end=block_address+old_block_used;
    // shift the blocks
    if(FCB_file_size!=new_size){
      u32 block_shift_num;
      u32 new_block_used=new_size/32;
      if(new_size%32!=0) new_block_used++;
      if(FCB_file_size>new_size){
      // 3)
        block_shift_num=old_block_used-new_block_used;
        // change FCB address
        for(int i=0;i<fs->FCB_ENTRIES;i++){
          // if valid
            if(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22]!=0xff){
              // if block address >= block_end, decrease by block_shift_num
              block_address=(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i+20]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i+21];
              if(block_address>=block_end){// decrease the FCB address by shift_num
                fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i+20]=(block_address-block_shift_num)/256;
                fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i+21]=(block_address-block_shift_num)%256;      
              }
            }
        }
        block_address=new_block_used;
        // shift blocks
        while(block_end!=empty_block_record){
          // loop until the empty block
          for(int j=0;j<32;j++){
            fs->volume[fs->FILE_BASE_ADDRESS+block_address*fs->STORAGE_BLOCK_SIZE+j]=fs->volume[fs->FILE_BASE_ADDRESS+block_end*fs->STORAGE_BLOCK_SIZE+j];
          }
          block_address++;
          block_end++;
        }
        // set bitmap
        // last used block=empty_block-1
        for (int i=empty_block_record-1;i>=empty_block_record-block_shift_num;i--){
          u32 bitmap_num=i/8;
          u32 bitmap_offest=i%8;
          fs->volume[bitmap_num]&=(~(1<<bitmap_offest));
          //fs->volume[i]=0;// invalid the block
        }
        // new_size=0-->RM
        // invalid the FCB
        if(new_size==0){
          fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22]=0xff;
        }
        // update the empty block pointer
        empty_block_record-=block_shift_num;      
      }else{
        //4)
        block_shift_num=new_block_used-old_block_used;
        if((empty_block_record+block_shift_num)<fs->STORAGE_SIZE){
        for(int i=0;i<fs->FCB_ENTRIES;i++){
          // if valid
            if(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22]!=0xff){
              // if block address >= block_end, decrease by block_shift_num
              block_address=(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i+20]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i+21];
              if(block_address>=block_end){// decrease the FCB address by shift_num
                fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i+20]=(block_address+block_shift_num)/256;
                fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i+21]=(block_address+block_shift_num)%256;      
              }
            }
        }
        block_address=empty_block_record-1;
        // shift blocks
        while(block_address!=block_end-1){
          // loop until the block_end
          for(int j=0;j<32;j++){
            fs->volume[fs->FILE_BASE_ADDRESS+(block_address+block_shift_num)*fs->STORAGE_BLOCK_SIZE+j]=fs->volume[fs->FILE_BASE_ADDRESS+block_address*fs->STORAGE_BLOCK_SIZE+j];
          }
          block_address--;
        }
        // set bitmap
        // last used block=empty_block-1
        for (int i=empty_block_record;i<empty_block_record+block_shift_num;i++){
          u32 bitmap_num=i/8;
          u32 bitmap_offest=i%8;
          fs->volume[bitmap_num]|=(1<<bitmap_offest);
         // fs->volume[i]=1;// invalid the block
        }  
        // update the empty block pointer
        empty_block_record+=block_shift_num;           
        }
        else{
          // simply return the old block address
          printf("ERROR, no enough space for allocating the new file");
        }
      }
    }
    //2)3)4)
      return (fs->volume[FCB_address+20]<<8)+fs->volume[FCB_address+21];
  }
}

__device__ void print_filename(FileSystem *fs, u32 fp){
  u32 ptr=fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp;
  while(fs->volume[ptr]!='\0'){
    printf("%c",fs->volume[ptr]);
    ptr++;
  }
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
  // 1) check fp valid or not
  // 2) check whether the directory is valid or not (bonus)
  // 3) find the block and read
    u32 FCB_address=fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp;
  if((fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22]!=0xff)&&(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 29]==0xff)){
    // the file is valid
    u32 block_address=(fs->volume[FCB_address+20]<<8)+fs->volume[FCB_address+21];
    for(int i=0;i<size;i++){
      output[i]=fs->volume[fs->FILE_BASE_ADDRESS+block_address*fs->STORAGE_BLOCK_SIZE+i%32];
      if(i%32==31) block_address++;
    }
  }else{
    printf("ERROR! file descripter is invalid for fs_read\n");
  }

  
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
  //1) check the file is valid and is not a folder
  //2) if valid, get the block address to start writing
  //3) update FCB, including: size, modified time, file type (bonus), parent directory(bonus)
  bool block_expand=false;
  //printf("fp: %u\n",fp);
  if((fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22]!=0xff)&&(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 29]==0xff)){
    // get a available block
    u32 block_address=fs_allocate_blocks(fs, size, fp);
    if(block_address==empty_block_record) block_expand=true;
  // set the block address
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 20]=block_address/256;
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 21]=block_address%256;
    // start writing
    for(int i=0;i<size;i++){
      // set the bitmap
      if(i%32==0){
        u32 bitmap_num=block_address/8;
        u32 bitmap_offest=block_address%8;
        fs->volume[bitmap_num]|=(1<<bitmap_offest);
      // empty block++ for the next file, even though it can be some space left
      if(block_expand) empty_block_record++;
      }
      fs->volume[fs->FILE_BASE_ADDRESS+block_address*fs->STORAGE_BLOCK_SIZE+i%32]=input[i];
      if(i%32==31){
      // go to the next block
        block_address++;
    //  printf("block_address: %u\n",block_address);

    //  printf("empty_block_record: %u\n",empty_block_record);
      }
    }
  // update FCB, including: size, modified time, file type (bonus), parent directory(bonus)
  // set file size
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 22]=size/(1<<16);
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 23]=(size%(1<<16))/(1<<8);
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 24]=size%(1<<8);
  // set modified time
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 27]=gtime/256;
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 28]=gtime%256; 
    gtime++;
 //   printf("gtime: %u\n",gtime);
  // set file_type
    fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fp + 29]=0xff;
  // set the parent directory
   // printf("empty_record: %u\n",empty_block_record);

  }else{
    printf("ERROR! the file descriptor for fs_write is invalid\n");
  }
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
  if(op==LS_D){
    printf("===sort by modified time===\n");
    u32 modified_time;
    // for-loop gtime(until all file in the dir are found) * FCB_entries times
    // check whether the file is in current directory (bonus)
    // a count variable is needed in (bonus)
    for(int i=gtime-1;i>=0;i--){
      for(int j=0;j<fs->FCB_ENTRIES;j++){
        //check valid bit
        if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 22]!=0xff){
          // the file is in the current directory(bonus) and is valid
          // check the modified time
          modified_time=(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 27]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 28];
          if(modified_time==i){
             print_filename(fs,j);
             printf("\n");
             break;
          }
        }
      }
    }
  }
  else{
    bool chose=false;
    // sort by size
    printf("===sort by file size===\n");
    // initilize the last_max as the larget file size which is 1024KB
    u32 last_max=(1<<20);
    // last_create time
    u32 last_create=0;
    // variable to record the largest file size in current loop
    u32 current_max=0;
    // variable to record the largest file index in current loop
    u32 current_max_index=0;
    // 2 temp record
    u32 current_size;
    u32 create_time;
    u32 last_create_index=0;
    u32 special_max=0xffff;
    for(int i=gtime-1;i>=0;i--){
      for(int j=0;j<fs->FCB_ENTRIES;j++){
        if (fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 22]!=0xff){
          // the file is in the current directory(bonus) and is valid
          // get the file size
          current_size=(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 22]<<16)+(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 23]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 24];
          // skip already printed items         
          if(current_size<=last_max){
              chose=true;
            if(current_size>current_max){
              current_max=current_size;
              current_max_index=j;
            }
            else if(current_size==current_max){

              create_time=(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * current_max_index + 25]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * current_max_index + 26]+1;
              if(create_time>((fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 25]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 26]+1)){
                if(((fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 25]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 26]+1)>last_create) {
                  current_max_index=j;
                }
              }else{
                if((create_time<=last_create)&&(((fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_create_index + 22]<<16)+(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_create_index + 23]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * last_create_index + 24])==current_size)) {
                  current_max_index=j;
                  continue;
                }
                if(current_size==0){
                  if(special_max>=(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 25]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 26]+1){
                      current_max_index=j;
                      special_max=(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 25]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * j + 26]+1;
                  }
                }
              }
            }
          }
        }
      }
      // after the loop, get the file with laregest file size among the remaining file
      if((current_max<last_max)&&chose){
          print_filename(fs,current_max_index);
          printf(" %u\n",current_max);
          last_max=current_max;
          //if(i==gtime-1)//last_create=0;
          last_create=(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * current_max_index + 25]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * current_max_index + 26]+1;
          last_create_index=current_max_index;
          //printf("last_create %u\n",last_create);

      }
      else if((current_max==last_max)&&chose){
      // in case there are files with the same file size
      if(last_create!=((fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * current_max_index + 25]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * current_max_index + 26]+1)){
        //printf("last_create %u\n",last_create);
        //printf("current_create %u\n",(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * current_max_index + 25]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * current_max_index + 26]+1);      
        last_create=(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * current_max_index + 25]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * current_max_index + 26]+1;
        last_create_index=current_max_index;
        print_filename(fs,current_max_index);
        printf(" %u\n",current_max);
        //print_filename(fs,current_max_index);
        //printf(" %u\n",current_max);
        //printf("last_create %u\n",last_create);
      }
      // to prevent stucking
      else{
        last_max-=1;
      }
      }
      current_max=0;
      current_max_index=0;
      chose=false;
    }
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  // find the name of the first file
  uchar* fcb_name=fs->volume+(fs->SUPERBLOCK_SIZE);
  u32 fp=fs->FCB_ENTRIES;
  u32 create_time;
  u32 modified_time;
  u32 new_create_time;
  u32 new_modified_time;
  for(int i=0;i<fs->FCB_ENTRIES;i++){
  // check valid bit and file_type
  if((fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22]!=0xff)&&(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 29]==0xff)){
  // valid
    if(is_file_found(s, fcb_name)){
      fp=i;// return the inumber if the file is found
      create_time=(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 25]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 26];
      modified_time=(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 27]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28];
      break;
    }
    // if the filename doesn't match
  }
    fcb_name+=32;
  }
  if(fp==fs->FCB_ENTRIES){
    printf("ERROR! file to be deleted cannot be found!\n");
    return;
  }
  if(op==RM){
    // FCB allocate
    fs_allocate_blocks(fs,0,fp);
  // update FCB_create time and FCB_modified_time, to increase the speed in LS_D and LS_S
  for(int i=0;i<fs->FCB_ENTRIES;i++){
  // check valid bit and file_type
  if((fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 22]!=0xff)&&(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 29]==0xff)){
      if(create_time<((fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 25]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 26])){
        new_create_time=(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 25]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 26];
        new_create_time-=1;
        fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 25]=new_create_time/256;
        fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 26]=new_create_time%256;
      }
      if(modified_time<((fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 27]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28])){
        new_modified_time=(fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 27]<<8)+fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28];
        new_modified_time-=1;
        fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 27]=new_modified_time/256;
        fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * i + 28]=new_modified_time%256;
      }
  }
  }
 // printf("gtime: %u\n",gtime);
  }
}
