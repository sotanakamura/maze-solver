/* Rumelhart, David E. et al. “Learning internal representations by error propagation.” (1986). */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define INPUT_DIM 2
#define HIDDEN_DIM 2
#define OUTPUT_DIM 1
#define LEARNING_RATE 0.03

double w_out_hidden[OUTPUT_DIM][HIDDEN_DIM];
double w_hidden_in[HIDDEN_DIM][INPUT_DIM];
double d_w_out_hidden[OUTPUT_DIM][HIDDEN_DIM];
double d_w_hidden_in[HIDDEN_DIM][INPUT_DIM];
double diff_output[OUTPUT_DIM];
double diff_hidden[HIDDEN_DIM];
double out[OUTPUT_DIM];
double out_hidden[HIDDEN_DIM];
double out_in[INPUT_DIM];

const double input[4][2] = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1},
};
const double dout[4] = {0, 1, 1, 0};

double sigmoid(double x)
{
    return 1.0 / (1 + exp(-x));
}

int main()
{
    srand(time(0));
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            w_out_hidden[i][j] = -1 + 2.0 * rand() / RAND_MAX;
        }
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            w_hidden_in[i][j] = -1 + 2.0 * rand() / RAND_MAX;
        }
    }
    
    for (int k = 0; k < 1000000; k++) {
        for (int i = 0; i < OUTPUT_DIM; i++) {
            for (int j = 0; j < HIDDEN_DIM; j++) {
                d_w_out_hidden[i][j] = 0;
            }
        }
        for (int i = 0; i < HIDDEN_DIM; i++) {
            for (int j = 0; j < INPUT_DIM; j++) {
                d_w_hidden_in[i][j] = 0;
            }
        }
    
        double e = 0;
    
        for (int c = 0; c < 4; c++) {
            for (int i = 0; i < INPUT_DIM; i++) {
                out_in[i] = sigmoid(input[c][i]);
            }
            for (int j = 0; j < HIDDEN_DIM; j++) {
                double total_input_j = 0;
                for (int i = 0; i < INPUT_DIM; i++) {
                    total_input_j += out_in[i] * w_hidden_in[j][i];
                }
                out_hidden[j] = sigmoid(total_input_j);
            }
            for (int j = 0; j < OUTPUT_DIM; j++) {
                double total_input_j = 0;
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    total_input_j += out_hidden[i] * w_out_hidden[j][i];
                }
                out[j] = sigmoid(total_input_j);
            }
            
            if (k % 1000 == 0)
                printf("input: {%1.0lf, %1.0lf}, output: %1.0lf\n", input[c][0], input[c][1], out[0]);
            e += 1.0 / 2 * pow(out[0] - dout[c], 2);
    
            for (int i = 0; i < OUTPUT_DIM; i++) {
                diff_output[i] = 0;
            }
            for (int i = 0; i < HIDDEN_DIM; i++) {
                diff_hidden[i] = 0;
            }
    
            for (int j = 0; j < OUTPUT_DIM; j++) {
                diff_output[j] = out[j] - dout[c];
            }
            for (int j = 0; j < OUTPUT_DIM; j++) {
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    d_w_out_hidden[j][i] += -LEARNING_RATE * diff_output[j] * out[j] * (1 - out[j]) * out_hidden[i];
                }
            }
            for (int i = 0; i < HIDDEN_DIM; i++) {
                for (int j = 0; j < OUTPUT_DIM; j++) {
                    diff_hidden[i] += diff_output[j] * out[j] * (1 - out[j]) * w_out_hidden[j][i];
                }
            }
            for (int j = 0; j < HIDDEN_DIM; j++) {
                for (int i = 0; i < INPUT_DIM; i++) {
                    d_w_hidden_in[j][i] += -LEARNING_RATE * diff_hidden[j] * out_hidden[j] * (1 - out_hidden[j]) * out_in[i];
                }
            }
        }
    
        for (int i = 0; i < OUTPUT_DIM; i++) {
            for (int j = 0; j < HIDDEN_DIM; j++) {
                w_out_hidden[i][j] += d_w_out_hidden[i][j];
            }
        }
        for (int i = 0; i < HIDDEN_DIM; i++) {
            for (int j = 0; j < INPUT_DIM; j++) {
                w_hidden_in[i][j] += d_w_hidden_in[i][j];
            }
        }
        
        if (k % 1000 == 0)
            printf("error: %lf\n", e);
    }

}