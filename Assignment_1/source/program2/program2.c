#include <linux/err.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/types.h>

MODULE_LICENSE("GPL");
extern pid_t kernel_clone(struct kernel_clone_args *kargs);
extern int do_execve(struct filename *filename,
		     const char __user *const __user *__argv,
		     const char __user *const __user *__envp);
extern pid_t kernel_threads(int (*fn)(void *), void *arg, unsigned long flags);

struct task_struct *p;

struct wait_opts {
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;
	struct waitid_info *wo_info;
	int wo_stat;
	struct rusage *wo_rusage;

	wait_queue_entry_t child_wait;
	int notask_error;
};
extern long do_wait(struct wait_opts *wo);
extern struct filename *getname_kernel(const char *filename);

static void my_wait(pid_t pid)
{
	int status;
	struct wait_opts wo;
	struct pid *wo_pid = NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);
	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WEXITED | WSTOPPED;
	wo.wo_info = NULL;
	wo.wo_stat = (int __user)status;
	wo.wo_rusage = NULL;
	int a;
	a = do_wait(&wo);
	if ((wo.wo_stat & 0x7f) == 127) {
		printk("[program2] : get SIGSTOP signal\n");
		printk("[program2] : child process terminated\n");
		printk("[program2] : The return signal is %d\n",
		       (wo.wo_stat >> 8) & 0xff);
	} else {
		switch (wo.wo_stat & 0x7f) {
		case 0: {
			printk("[program2] : Program executes normally\n");
			printk("[program2] : get SIGCHLD signal\n");
			break;
		}
		case 1: {
			printk("[program2] : get SIGHUP signal\n");
			break;
		}
		case 2: {
			printk("[program2] : get SIGINT signal\n");
			break;
		}
		case 3: {
			printk("[program2] : get SIGQUIT signal\n");
			break;
		}
		case 4: {
			printk("[program2] : get SIGILL signal\n");
			break;
		}
		case 5: {
			printk("[program2] : get SIGTRAP signal\n");
			break;
		}
		case 6: {
			printk("[program2] : get SIGABRT signal\n");
			break;
		}
		case 7: {
			printk("[program2] : get SIGBUS signal\n");
			break;
		}
		case 8: {
			printk("[program2] : get SIGFPE signal\n");
			break;
		}
		case 9: {
			printk("[program2] : get SIGKILL signal\n");
			break;
		}
		case 10: {
			printk("[program2] : get SIGUSR1 signal\n");
			break;
		}
		case 11: {
			printk("[program2] : get SIGSEGV signal\n");
			break;
		}
		case 12: {
			printk("[program2] : get SIGUSR2 signal\n");
			break;
		}
		case 13: {
			printk("[program2] : get SIGPIPE signal\n");
			break;
		}
		case 14: {
			printk("[program2] : get SIGALRM signal\n");
			break;
		}
		case 15: {
			printk("[program2] : get SIGTERM signal\n");
			break;
		}
		case 16: {
			printk("[program2] : get SIGSTKFLT signal\n");
			break;
		}
		case 17: {
			printk("[program2] : get SIGCHLD signal\n");
			break;
		}
		case 18: {
			printk("[program2] : get SIGCONT signal\n");
			break;
		}
		case 19: {
			printk("[program2] : get SIGSTOP signal\n");
			break;
		}
		case 20: {
			printk("[program2] : get SIGTSTP signal\n");
			break;
		}
		case 21: {
			printk("[program2] : get SIGTTIN signal\n");
			break;
		}
		case 22: {
			printk("[program2] : get SIGTTOU signal\n");
			break;
		}
		case 23: {
			printk("[program2] : get SIGURG signal\n");
			break;
		}
		case 24: {
			printk("[program2] : get SIGXCPU signal\n");
			break;
		}
		case 25: {
			printk("[program2] : get SIGXFSZ signal\n");
			break;
		}
		case 26: {
			printk("[program2] : get SIGVTALRM signal\n");
			break;
		}
		case 27: {
			printk("[program2] : get SIGPROF signal\n");
			break;
		}
		case 28: {
			printk("[program2] : get SIGWINCH signal\n");
			break;
		}
		case 29: {
			printk("[program2] : get SIGIO signal\n");
			break;
		}
		case 30: {
			printk("[program2] : get SIGPWR signal\n");
			break;
		}
		case 31: {
			printk("[program2] : get SIGSYS signal\n");
			break;
		}
		}
		printk("[program2] : child process terminated\n");
		printk("[program2] : The return signal is %d\n",
		       wo.wo_stat & 0x7f);
	}
	put_pid(wo_pid);
	return;
}

static int my_exec(void *argc)
{
	printk("[program2] : child process\n");
	int a;
	a = do_execve(getname_kernel("/tmp/test"),
		      NULL, NULL);
	return a;
}

// implement fork function
static int my_fork(void *argc)
{
	// set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for (i = 0; i < _NSIG; i++) {
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	printk("[program2] : module_init kthread start\n");
	/* execute a test program in child process */
	pid_t pid = kernel_thread(&my_exec, 0, SIGCHLD);
	printk("[program2] : The child process has pid=%d\n", pid);
	printk("[program2] : This is the parent process, pid= %d\n",
	       current->pid);
	/* wait until child process terminates */
	my_wait(pid);
	return 0;
}

static int __init program2_init(void)
{
	printk("[program2] : module_init {Tao Chujun} {120090211}\n");
	/* create a kernel thread to run my_fork */
	p = kthread_create(&my_fork, NULL, "kthread created");
	printk("[program2] : module_init create kthread start\n");
	if (p != NULL) {
		wake_up_process(p);

	} else {
		printk("error");
	}

	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : module_exit./my\n");
}

module_init(program2_init);
module_exit(program2_exit);
