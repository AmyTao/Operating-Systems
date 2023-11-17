#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <curses.h>


struct pinfo {
	int pid;
	int ppid;
	char name[50];
	int depth;
	struct pinfo *child;
	struct pinfo *bro;
};

struct pinfo processINFOs[500];
int number_process = 0;
int getPPID(char *filename);
void setpid_ppid();
void getName(char *filename, char *name);
int recursive_print_tree(WINDOW *win, struct pinfo *head, int p_x,int p_y);
void create_list(struct pinfo *processINFOs);
bool recursive_find(struct pinfo *ptr,struct pinfo *target);
void ADD(struct pinfo *parent,struct pinfo *ch);



int main()
{
	setpid_ppid();
/* 	for (int i = 0; i < number_process; i++) {
		printf("%d %d \n", processINFOs[i].pid, processINFOs[i].ppid);
	}
 */	
	create_list(processINFOs);
	//sleep(5);
	
    WINDOW *ptr=initscr();
    move(0,0);
	recursive_print_tree(ptr,&processINFOs[0],0,0);
    refresh();
    sleep(100);
    endwin();
    exit(0);
	return 0;
}

int getPPID(char *filename)
{
	int ppid = -100;
	char *right = NULL;
	FILE *fp = fopen(filename, "r");
	char info[51];
	info[50] = '\0';
	if (fp == NULL) {
		printf("error\n");
		return -1;
	}
	if (fgets(info, 50, fp) == NULL) {
		puts("fgets error!");
		exit(0);
	}
	right = strchr(info, ')');
	if (right == NULL) {
		printf("not find\n");
	}
	if (right[0] != ')')
		right += 3;
	else {
		right += 4;
	}
	sscanf(right, "%d", &ppid);
	return ppid;
}

void setpid_ppid()
{
	DIR *dir_ptr;
	struct dirent *direntp;
	int pid;
	int ppid;
	char process_path[51] = "/proc/";
	char stat[6] = "/stat";
	char pidstr[20];

	dir_ptr = opendir("/proc");
	if (dir_ptr == NULL) {
		fprintf(stderr, "cannot open /proc\n");
		exit(0);
	}
	//add init 0 process
	/*
	processINFOs[number_process].pid = 0;
	processINFOs[number_process].ppid =0;
	processINFOs[number_process].bro=NULL;
	processINFOs[number_process].depth=0;
	processINFOs[number_process].child=NULL;
	char* firstname="init";
	char an[50];
	memset(an, 0, sizeof(an));
	strncpy(an, firstname, 5);
	for (int i = 0; i < 50; i++) {
		processINFOs[number_process].name[i] = an[i];
	}

	number_process++;*/


	while (direntp = readdir(dir_ptr)) {
		pid = atoi(direntp->d_name);
		if (pid != 0) {
			processINFOs[number_process].pid = pid;
			sprintf(pidstr, "%d", pid);
			strcat(process_path, pidstr);
			strcat(process_path, stat);
			int ppid = getPPID(process_path);
			if((ppid!=2)&&(pid!=2)){
				if (ppid != -1) {			
					processINFOs[number_process].ppid =
						ppid;
					getName(process_path,
						processINFOs[number_process]
							.name);
					printf("%s  :pid : %d:ppid: %d\n",processINFOs[number_process].name,pid,ppid);
					processINFOs[number_process].child =
						NULL;
					processINFOs[number_process].bro = NULL;
					processINFOs[number_process++].depth =
						0;
				
			} else
				number_process++;		
		}
		process_path[6] = 0;
	}
}
}
void getName(char *filename, char *name)
{
	char *left = NULL;
	char *right = NULL;
	FILE *fp = fopen(filename, "r");
	char info[51];
	info[50] = '\0';
	right = strchr(info, ')');
	right++;
	left = strchr(info, '(');
	int answer = right - left;
	char ans[50];
	memset(ans, 0, sizeof(ans));
	strncpy(ans, left, answer);
	for (int i = 0; i < 50; i++) {
		name[i] = ans[i];
	}
	return;
}

bool recursive_find(struct pinfo *ptr,struct pinfo *target){
	if(ptr==NULL){
		return false;
	}
	else{
		if(ptr->pid==target->ppid){
			ADD(ptr,target);
			return true;
		}else{
			if(ptr->bro!=NULL){
				bool Bro=recursive_find(ptr->bro,target);
				if(Bro) return true;
			}
			if(ptr->child!=NULL){
				bool Child=recursive_find(ptr->child,target);
				if(Child) return true;
			}
			return false;
		}

	}
}

void ADD(struct pinfo *parent,struct pinfo *ch){
			printf("parent: %s  :pid : %d:ppid: %d\n",parent->name,parent->pid,parent->ppid);
			printf("child: %s  :pid : %d:ppid: %d\n",ch->name,ch->pid,ch->ppid);
			parent->depth++;
			if (parent->child == NULL) {
				parent->child = ch;
			} else {
				if (parent->child->bro != NULL) {
					parent = parent->child;
					while (parent->bro != NULL) {
						if ((parent->pid <
						     ch->pid) &&
						    (parent->bro->pid <
						     ch->pid)) {
							parent = parent->bro;
						} else {
							ch->bro =
								parent->bro;
							parent->bro =
								ch;
							break;
						}
					}
					parent->bro = ch;
				} else {
					if (parent->child->pid <
					    ch->pid) {
						parent->child->bro =
							ch;
					} else {
						ch->bro =
							parent->child;
						parent->child = ch;
					}
				}
			}

}

void create_list(struct pinfo *processINFOs)
{
	struct pinfo* waiting_list[100];
	int num_of_waiting=0;
	for (int i = 1; i < number_process; i++) {
		if(processINFOs[i].ppid==processINFOs[0].pid)
			ADD(&processINFOs[0],&processINFOs[i]);
		else{
			waiting_list[num_of_waiting]=&processINFOs[i];
			num_of_waiting++;
		}
	}
	bool is_find;
	int count=0;
	while(num_of_waiting!=0|count<=3){
		for(int i=0;i<num_of_waiting;i++){
			printf("%s  :pid : %d:ppid: %d\n",waiting_list[i]->name,waiting_list[i]->pid,waiting_list[i]->ppid);
			is_find=recursive_find(&processINFOs[0],waiting_list[i]);
			if(is_find){
				for(int j=i;j<num_of_waiting-1;j++){
					waiting_list[j]=waiting_list[j+1];
				}
				num_of_waiting--;
				i--;
			}
		}
		count++;

	}
	return;
}

int recursive_print_tree(WINDOW *win, struct pinfo *head, int p_x,int p_y)
{
	int depth=0;
	int cur_x,cur_y;
	if ((head->child == NULL) && (head->bro == NULL)) {
		printw("_%s %d", head->name,head->pid);
		depth=0;
	} else if (head->child == NULL && head->bro != NULL) {
		printw("_%s %d", head->name,head->pid);
		move(p_y+1,p_x-1);
		printw("|");
		depth=recursive_print_tree(win,head->bro,p_x,p_y+1)+1;
	} else if (head->child != NULL && head->bro == NULL) {
		printw("_%s %d_ ", head->name,head->pid);
		cur_x=getcurx(win);
		depth=recursive_print_tree(win,head->child,cur_x,p_y);
	} else {
		printw("_%s %d ", head->name,head->pid);
		cur_x=getcurx(win);
		depth=recursive_print_tree(win,head->child,cur_x,p_y);
		move(p_y,p_x);
		cur_x=p_x;
		cur_y=p_y;
		for(int i=0;i<=depth;i++){
			move(--cur_x,--cur_y);
			printw("|");
		}
		getsyx(cur_y,cur_x);
		depth=recursive_print_tree(win,head->bro,cur_x,cur_y);

	}
	move(p_y,p_x);
	return depth;
}
