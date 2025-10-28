include <iostream>
using namespace std;

float sum (float a, float b) {
  return a + b;
}


float subtraction (float a, float b){
  return a + b;
}

float division (float a, float b){
  if (b == 0):{
    cout << "Erro. Impossível dividir por zero." << endl;
  }

return a/b;

}

int main(){
float x, y;
char op;
  cout<<"Calculadora básica"<<endl;
