#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

class Dense {
    public:
        float W;
        float B;
        Dense() {
            W = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            B = 0;
        }
        float Call(float x) {
            return W * x + B;
        }
        void Optimize(float w, float b) {
            W -= w;
            B -= b;
        }
};

float* InputData(int stop, int quantity) {
    float* input[quantity];
    for (int i = 0; i <= quantity; i++) {
        input[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    return input;
}
float* OutputData(float *input, int quantity) {
    float* output[quantity];
    for (int i = 0 ; i <=quantity; i++){
        output[i] = 2.0 * input[i] + 3.0;
    }
    return output;
}

void printArray(float *list) {
    for (int i = 0 ; i <=2; i++){
        cout << list[i] <<"\n";
    }
}

int main() {
    Dense dense;
    float* input = InputData(10, 2);
    float* output = OutputData(input, 2);

    cout << dense.W  << "\n";
    dense.Optimize(0.1, 0.2);
    cout << dense.W << "\n";
    return 0;
}