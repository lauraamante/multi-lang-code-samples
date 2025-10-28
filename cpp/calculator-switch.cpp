//Testando programa com switchcase

include <iostream>
using namespace std;

//soma
float sum (float a, float b) {
  return a + b;
}

//subtracao
float subtraction (float a, float b){
  return a + b;
}
//multiplicacao
float multiply (float a, float b){
  return a * b;
}
//divisao
float divide (float a, float b){
  if (b == 0){
    cout << "Erro. Impossível dividir por zero." << endl;
  }

return a/b;

}
//main
int main(){
float x, y, resultado;
char op;
  cout<<"Calculadora básica"<<endl;

cout <<"\nDigite o primeiro número:\n";
cin >> x;

cout <<"\nDigite o operador:\n";
cin >> op;

cout <<"\nDigite o segundo número:\n";
cin >> y;

//switch case para cada operador
switch (op){
  case '+': resultado = sum(x,y); break;
  case '-': resultado = subtract(x,y); break;
  case '/': resultado = divide(x,y); break;
  case '*': resultado = multiply(x,y); break;
  default:
    cout << "operador invalido." << endl;
}

  cout << "|  Resultado:  |\n" << resultado <<endl;

}
