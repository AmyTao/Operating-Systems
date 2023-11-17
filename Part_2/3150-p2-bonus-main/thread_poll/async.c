
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"
#include <string.h>
#include <stdio.h>



void* routine(void*arg){
    //the third argument passing to pthread_create
    struct my_worker* w=(struct my_worker*)arg;//point to the worker thread
    while(1){
        pthread_mutex_lock(&thread_pool.task_queue_lock);//get access to the task queue
        while(thread_pool.task_queue.size==0){//empty task queue
            pthread_cond_wait(&thread_pool.job_signal,&thread_pool.task_queue_lock);//wait for signal, unlock task queue lock
        }//get the signal
        struct my_item *run_item=thread_pool.task_queue.head;//get the task
        DL_DELETE(thread_pool.task_queue.head, thread_pool.task_queue.head);//remove task from the task queue
        thread_pool.task_queue.size--;//delete task queue size
        pthread_mutex_unlock(&thread_pool.task_queue_lock);//unlock the task queue lock
        run_item->func(run_item->argument);//run the job
        free(run_item);//free the job
    }
        free(w);//free the worker
        pthread_exit(NULL);
}

//./httpserver --files files/ --port 8000 --num-threads T
//ab -n X -c T http://localhost:8000/


void async_init(int num_threads) {
   
    /** TODO: create num_threads threads and initialize the thread pool **/
    //initiate the locks
    pthread_cond_init(&thread_pool.job_signal,NULL);
    pthread_mutex_init(&thread_pool.task_queue_lock,NULL);
    //initiate the threadpool
    memset(&thread_pool,0,sizeof(struct ThreadPool));
    thread_pool.threads.size=0;
    thread_pool.task_queue.size=0;
    thread_pool.task_queue.head=NULL;
    thread_pool.threads.head=NULL;
    for(int i=0;i<num_threads;i++){//add num_threads worker to the thread pool
        //create a worker
        struct my_worker *w=(struct my_worker*)malloc(sizeof(struct my_worker));
        //initialize the worker
        memset(w,0,sizeof(struct my_worker));
        //set the worker 
        w->is_stop=0;
        pthread_t Tid;
        //append the worker to the threads list
        DL_APPEND(thread_pool.threads.head, w);
        //add the threads size
        thread_pool.threads.size++;
        //create a pthread which runs the "routine" function with w as the passing argument
        pthread_create(&Tid,NULL,routine,(void*)w);
         w->tid=&Tid;
}
    return;
}
void async_run(void (*hanlder)(int), int args) {
    /** TODO: rewrite it to support thread pool **/
    //create a item
    struct my_item *new_item=(struct my_item*)malloc(sizeof(struct my_item));
    memset(new_item,0,sizeof(struct my_item));
    new_item->func=hanlder;
    new_item->argument=args;
    //get the task queue lock
    pthread_mutex_lock(&thread_pool.task_queue_lock);
    //append the job to task queue
    DL_APPEND(thread_pool.task_queue.head, new_item);
    //add the task queue size
    thread_pool.task_queue.size++;
    pthread_cond_signal(&thread_pool.job_signal);
    pthread_mutex_unlock(&thread_pool.task_queue_lock);    
    return;
}