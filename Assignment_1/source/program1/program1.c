#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[], char **environ)
{
	printf("Process start to fork\n");
	/* fork a child process */
	pid_t return_pid = fork();
	int status;
	/* execute test program */
	if (return_pid < 0)
		printf("error in fork");
	else if (return_pid == 0) {
		printf("I'm the child process, my pid = %d \n", getpid());
		printf("Child process start to execute test program\n");
		execve(argv[1], argv, environ);
	} else {
		printf("I'm the parent process, my pid = %d \n", getpid());
		/* wait for child process terminates */
		pid_t w = waitpid(return_pid, &status, WUNTRACED | WCONTINUED);
		int i = WEXITSTATUS(status);
		/* check child process'  termination status */
		printf("Parent process receving the SIGCHLD signal\n");
		if (WIFEXITED(status)) {
			printf("Normal termination with EXIT STATUS=%d\n",
			       WEXITSTATUS(status));
		} else if (WIFSIGNALED(status)) {
			switch (WTERMSIG(status)) {
			case 1:
				printf("Parent process receives SIGHUP signal\n");
				break;
			case 2:
				printf("Parent process receives SIGINT signal\n");
				break;
			case 3:
				printf("Parent process receives SIGQUIT signal\n");
				break;
			case 4:
				printf("Parent process receives SIGILL signal\n");
				break;
			case 5:
				printf("Parent process receives SIGTRAP signal\n");
				break;
			case 6:
				printf("Parent process receives SIGABRT signal\n");
				break;
			case 7:
				printf("Parent process receives SIGBUS signal\n");
				break;
			case 8:
				printf("Parent process receives SIGFPE signal\n");
				break;
			case 9:
				printf("Parent process receives SIGKILL signal\n");
				break;
			case 10:
				printf("Parent process receives SIGUSR1 signal\n");
				break;
			case 11:
				printf("Parent process receives SIGSEGV signal\n");
				break;
			case 12:
				printf("Parent process receives SIGUSR2 signal\n");
				break;
			case 13:
				printf("Parent process receives SIGPIPE signal\n");
				break;
			case 14:
				printf("Parent process receives SIGALRM signal\n");
				break;
			case 15:
				printf("Parent process receives SIGTERM signal\n");
				break;
			case 16:
				printf("Parent process receives SIGSTKFLT signal\n");
				break;
			case 17:
				printf("Parent process receives SIGCHLD signal\n");
				break;
			case 18:
				printf("Parent process receives SIGCONT signal\n");
				break;
			case 19:
				printf("Parent process receives SIGSTOP signal\n");
				break;
			case 20:
				printf("Parent process receives SIGTSTP signal\n");
				break;
			case 21:
				printf("Parent process receives SIGTTIN signal\n");
				break;
			case 22:
				printf("Parent process receives SIGTTOU signal\n");
				break;
			case 23:
				printf("Parent process receives SIGURG signal\n");
				break;
			case 24:
				printf("Parent process receives SIGXCPU signal\n");
				break;
			case 25:
				printf("Parent process receives SIGXFSZ signal\n");
				break;
			case 26:
				printf("Parent process receives SIGVTALRM signal\n");
				break;
			case 27:
				printf("Parent process receives SIGPROF signal\n");
				break;
			case 28:
				printf("Parent process receives SIGWINCH signal\n");
				break;
			case 29:
				printf("Parent process receives SIGIO signal\n");
				break;
			case 30:
				printf("Parent process receives SIGPWR signal\n");
				break;
			case 31:
				printf("Parent process receives SIGSYS signal\n");
				break;
			default:
				printf("OTHER SIGNALS\n");
			}
		} else if (WIFSTOPPED(status)) {
			printf("Child process get SIGSTOP signal %d\n",
			       WSTOPSIG(status));
		} else {
			printf("CHILD PROCESS CONTINUE\n");
		}
		exit(0);
	}
	return 0;
}
