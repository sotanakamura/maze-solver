#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAZE_SIZE 4
#define GOALX 3
#define GOALY 3
#define MOVE_REWARD -1
#define GOAL_REWARD 10
#define HIT_REWARD -10
#define DISCOUNT_RATE 0.9
#define LERANING_RATE 0.1
#define STEP 1000000

char maze[MAZE_SIZE * 2 + 1][MAZE_SIZE * 4 + 2] = {
	"+---+---+---+---+",
	"|   |       |   |",
	"+   +   +   +   +",
	"|       |       |",
	"+---+   +   +---+",
	"|       |       |",
	"+   +   +---+   +",
	"|   |       |   |",
	"+---+---+---+---+",
};

double Q[MAZE_SIZE][MAZE_SIZE][4] = {0};
char x = 0;
char y = 0;

enum direction
{
	LEFT, UP, DOWN, RIGHT
};

int main()
{
	/* learning */
	for(int step = 0; step < STEP; step++) {
		/* take an action */
		double num = (double)rand() / RAND_MAX;
		double prob[4];
		double sum = 0;
		char action = 0;
		for (int a = 0; a < 4; a++) {
			sum += exp(Q[y][x][a]);
		}
		for (int a = 0; a < 4; a++) {
			/* softmax */
			prob[a] = exp(Q[y][x][a]) / sum;
		}
		for (int a = 1; a < 4; a++) {
			/* roulette */
			prob[a] += prob[a - 1];
		}
		for (int a = 0; a < 4; a++) {
			if (num < prob[a]) {
				action = a;
				break;
			}
		}

		/* move and get a reward */
		char pre_x = x;
		char pre_y = y;
		char reward = 0;
		switch (action)
		{
		case LEFT:
			if (maze[y * 2 + 1][x * 4 + 2 - 2] == ' ') {
				x--;
				reward = MOVE_REWARD;
			}
			break;

		case UP:
			if (maze[y * 2 + 1 - 1][x * 4 + 2] == ' ') {
				y--;
				reward = MOVE_REWARD;
			}
			break;

		case DOWN:
			if (maze[y * 2 + 1 + 1][x * 4 + 2] == ' ') {
				y++;
				reward = MOVE_REWARD;
			}
			break;

		case RIGHT:
			if (maze[y * 2 + 1][x * 4 + 2 + 2] == ' ') {
				x++;
				reward = MOVE_REWARD;
			}
			break;
		}
		if (x == GOALX && y == GOALY) {
			reward = GOAL_REWARD;
		}
		if (!reward) {
			reward = HIT_REWARD;
		}

		/* TD error */
		double maxq = 0;
		for (int a = 0; a < 4; a++) {
			if (maxq < Q[y][x][a]) {
				maxq = Q[y][x][a];
			}
		}
		double delta = (reward + DISCOUNT_RATE * maxq) 
					- Q[pre_y][pre_x][action];
		Q[pre_y][pre_x][action] += LERANING_RATE * delta;

		if (step < 50 || STEP - step < 50) {
			/* print the maze */
			maze[y * 2 + 1][x * 4 + 2] = 'O';
			puts("\033[?25l");
			puts("\033[0;0H");
			printf("step: %d\n", step);
			for (int i = 0; i < MAZE_SIZE * 2 + 1; i++) {
				puts(maze[i]);
			}
			maze[y * 2 + 1][x * 4 + 2] = ' ';
			
			/* sleep */
			for (int i = 0; i < 100000000; i++) {}
		}
		
		/* reset */
		if (x == GOALX && y == GOALY) {
			x = y = 0;
		}
	}

	/* print result */
	char title[4][10] = {"left", "up", "down", "right"};
	for (int a = 0; a < 4; a++) {
		puts("---");
		puts(title[a]);
		for (int i = 0; i < MAZE_SIZE; i++) {
			for (int j = 0; j < MAZE_SIZE; j++) {
				printf("%3lf, ", Q[i][j][a]);
			}
			puts("");
		}
	}
}