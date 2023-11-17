#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>

//job queue and items
typedef struct my_item {
  /* TODO: More stuff here, maybe? */
  struct my_item *next;
  struct my_item *prev;
  void (*func) (int);
  int argument;
} my_item_t;

typedef struct my_queue {
  int size;
  my_item_t *head;
  /* TODO: More stuff here, maybe? */
} my_queue_t;

typedef struct my_worker {//one working thread
      pthread_t *tid;
      int is_stop;
  struct my_worker *next;
  struct my_worker *prev;
}my_worker_t;

typedef struct my_threads{//a double linked list connecting all threads
  int size;
  my_worker_t *head;
}my_threads_t;


typedef struct ThreadPool{
    int size;
    pthread_cond_t job_signal;
    pthread_mutex_t task_queue_lock;
    my_queue_t task_queue;
    my_threads_t threads;
}my_TP_t;





struct ThreadPool thread_pool;



int adding_jobs();
void* routine(void*arg);//the third argument passing to pthread_create
void async_init(int);
void async_run(void (*fx)(int), int args);

#endif
