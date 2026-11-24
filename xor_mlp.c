#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// network topology: 2 inputs → 2 hidden → 1 output
float w_ih[2][2]; // input→hidden weights
float b_h[2];     // hidden biases
float w_ho[2];    // hidden→output weights
float b_o;        // output bias

// activation function
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// derivative of sigmoid
float dsigmoid(float y) {
    return y * (1.0f - y);
}

// initialize small random weights
void init_net() {
    for (int i=0;i<2;i++){
        for (int j=0;j<2;j++)
            w_ih[i][j] = ((rand()%2001)-1000)/5000.0f; // ~[-0.2,0.2]
        b_h[i] = 0.0f;
        w_ho[i] = ((rand()%2001)-1000)/5000.0f;
    }
    b_o = 0.0f;
}

// forward pass
float forward(float x1, float x2, float *h1, float *h2) {
    *h1 = sigmoid(w_ih[0][0]*x1 + w_ih[0][1]*x2 + b_h[0]);
    *h2 = sigmoid(w_ih[1][0]*x1 + w_ih[1][1]*x2 + b_h[1]);
    return sigmoid(w_ho[0]*(*h1) + w_ho[1]*(*h2) + b_o);
}

// one training step
void train_sample(float x1, float x2, float target, float lr) {
    float h1, h2;
    float y = forward(x1, x2, &h1, &h2);

    float error = target - y;
    float d_out = error * dsigmoid(y);

    float d_h1 = dsigmoid(h1) * w_ho[0] * d_out;
    float d_h2 = dsigmoid(h2) * w_ho[1] * d_out;

    // update hidden→output
    w_ho[0] += lr * d_out * h1;
    w_ho[1] += lr * d_out * h2;
    b_o      += lr * d_out;

    // update input→hidden
    w_ih[0][0] += lr * d_h1 * x1;
    w_ih[0][1] += lr * d_h1 * x2;
    b_h[0]     += lr * d_h1;

    w_ih[1][0] += lr * d_h2 * x1;
    w_ih[1][1] += lr * d_h2 * x2;
    b_h[1]     += lr * d_h2;
}

int main(void) {
    srand(1);
    init_net();
    float lr = 0.01f;

    float inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    float targets[4] = {0.0f, 1.0f, 1.0f, 0.0f};

    for (int epoch=0; epoch<50000000; epoch++) {
        for (int i=0;i<4;i++)
            train_sample(inputs[i][0], inputs[i][1], targets[i], lr);
    }

    printf("Trained XOR network:\n");
    for (int i=0;i<4;i++) {
        float h1,h2;
        float y = forward(inputs[i][0], inputs[i][1], &h1, &h2);
        printf("Input %.0f %.0f -> %.3f\n", 
               inputs[i][0], inputs[i][1], y);
    }
    return 0;
}
