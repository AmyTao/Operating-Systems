#include <unistd.h>
#include <stdio.h>
#include <signal.h>

int main(int argc,char* argv[]){
	int i=0;

	printf("--------USER PROGRAM--------\n");
		//sleep(2);
//	alarm(2);
	//raise(SIGBUS);
	//raise(SIGINT);
	//raise(SIGABRT);
	//raise(SIGALRM);
	//raise(SIGFPE);
	//raise(SIGHUP);
	//raise(SIGILL);
	//raise(SIGKILL);
	//raise(SIGPIPE);
	//raise(SIGQUIT);
	//raise(SIGSEGV);
	raise(SIGSTOP);
	//raise(SIGTERM);
	//raise(SIGTRAP);


	sleep(5);
	printf("user process success!!\n");
	printf("--------USER PROGRAM--------\n");
	return 100;
}
