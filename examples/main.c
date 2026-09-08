#include <stdio.h>


int main(void){
    double array[3] = {1.12371,2.13132,3.1};
    double sum;


    for (int i = 0; i < 3; i++){
        sum += array[i];
    }

    printf("%lf\n", sum);

    return 0;
}
