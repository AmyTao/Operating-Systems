#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include <cstdlib>
#include <iostream>
#include <ctime>

#define ROW 10
#define COLUMN 50 

using namespace std;

pthread_mutex_t mutex;


struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 

int status;
char map[ROW+10][COLUMN] ; 

double random(double start, double end)
{
    return start+(end-start)*rand()/(RAND_MAX + 1.0);
}

bool check_status(int x, int y){//true=end the game
	int next_x,last_x;
	if(x==0) return true;
	if(x==ROW) return false;
	if(y==0||y==COLUMN-1) return true;
	else{
		if (((map[x][y-1]!='=')&&(map[x][y+1]!='='))){
			return true;
		}
		return false;
	}
}

// Check the inputs. Return 1: change the position of the frog. 	2: quit the game 	0: No input 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		
		if(ch=='w'&&frog.x>0){
			frog.x--;
			return 1;
		}
		else if(ch=='s'&&frog.x<ROW){
			frog.x++;
			return 1;
		} 
		else if(ch=='a'&&frog.y>0){
			frog.y--;
			return 1;
		}
		else if(ch=='d'&&frog.y<COLUMN-1){
			frog.y++;
			return 1;
		}
		else{
			if (ch=='q'){
			return 2;
			}
		}
	}
	return 0;
}


void *logs_move( void *t ){
	int i,j,next;

	/*  Move the logs  */
	while(!(check_status(frog.x,frog.y))){
		pthread_mutex_lock(&mutex);
		for(i=1;i<=ROW-1;i++){
			if(i%2==1){
				next=map[i][49];
				for(j=COLUMN-1;j>=1;j--){
					map[i][j]=map[i][j-1];//odd move right

					if(map[i][j]=='0') {frog.y++;
					}
				}
				map[i][0]=next;

			}
			else{
				next=map[i][0];
				for(j=1;j<COLUMN;j++){
					map[i][j-1]=map[i][j];
					if(map[i][j-1]=='0') {frog.y--;
					}
				}
				map[i][COLUMN-1]=next;
			}
		}
		pthread_mutex_unlock(&mutex);
		if(status==2){
			break;
		}
		if(check_status(frog.x,frog.y)) {
			printf("check log");
			break;
		}
		for(int d=0;d<40000000;d++){
            
        }

	}
	pthread_exit(NULL);
}

void*frog_control(void*t){
	
	/*  Check keyboard hits, to change frog's position or quit the game. */
	
	int old_x,old_y,new_x,new_y,i;
	while(!(check_status(frog.x,frog.y))){
	
	old_x=frog.x;
	old_y=frog.y;

	status=kbhit();
	if(status==1){
		pthread_mutex_lock(&mutex);

		if(old_x==ROW){
		map[old_x][old_y]='|';
		}
		else if(frog.x==ROW) map[old_x][old_y]='=';
		else if(map[frog.x][frog.y-1]==' '&&map[frog.x][frog.y+1]==' ') map[old_x][old_y]='=';
		else{map[old_x][old_y]=map[frog.x][frog.y];}
		map[frog.x][frog.y]='0';
					system("clear");
	pthread_mutex_unlock(&mutex);

		for(int i=0;i<=ROW;i++){
			for(int j=0;j<COLUMN;j++){
				cout<<map[i][j];
			}
			printf("\n");
		}

			
	}

	if(status==2){
		break;
	}

	}
	pthread_exit(NULL);

}

void*print_map(void*t){
	while(!(check_status(frog.x,frog.y))){
			system("clear");
		for(int i=0;i<=ROW;i++){
			for(int j=0;j<COLUMN;j++){
				cout<<map[i][j];
			}
						printf("\n");

		}

		for(int d=0;d<10000000;d++){
            
        }
	if(status==2) break;		
	}
			system("clear");
		for(int i=0;i<=ROW;i++){
			for(int j=0;j<COLUMN;j++){
				cout<<map[i][j];
			}
			printf("\n");
		}
	pthread_exit(NULL);
}

int main( int argc, char *argv[] ){
	if(pthread_mutex_init(&mutex,NULL)!=0){
		printf("error\n");
	};
	double random(double,double);
	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 
	int position;
    srand(unsigned(time(0)));
	for(int i=1;i<10;i++){
		position=random(0,50);
		for(int j=0;j<15;j++){
			
			map[i][position]='=';
			position=(position+1)%50;
		}
	}

	//Print the map into screen
	printf("\033[?25l");
	for(i=0;i<=ROW;i++){
		for(int j=0;j<COLUMN;j++){
			cout<<map[i][j];
		}
		printf("\n");
	}


	/*  Create pthreads for wood move and frog control.  */
	
	int logs_thread,frog_thread,q,print_th;
	pthread_t logs,Frog,printer;
	logs_thread=pthread_create(&logs,NULL,logs_move,(void*)q);
	frog_thread=pthread_create(&Frog,NULL,frog_control,(void*)q);
	print_th=pthread_create(&printer,NULL,print_map,(void*)q);
	if(logs_thread!=0|frog_thread!=0) printf("error");

	pthread_join(logs,NULL);
	pthread_join(Frog,NULL);
	pthread_join(printer,NULL);

	/*  Display the output for user: win, lose or quit.  */
	
	system("clear");
	if(frog.x==0){
	printf("You win the game!\n");
	}
	else if(status==2){
	printf("You exit the game.\n");
	}

	else{
		
	printf("You lose the game!\n");

	}
	return 0;
}

