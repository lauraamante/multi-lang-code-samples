#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main() {
    srand(time(0)); // inicializa o gerador de números aleatórios
    int numero_secreto = rand() % 100 + 1; // número aleatório entre 1 e 100
    int tentativa;
    int tentativas = 0;

    cout << "Bem-vindo ao Jogo de Adivinhação!" << endl;
    cout << "Tente adivinhar o número entre 1 e 100." << endl;

    do {
        cout << "Digite sua tentativa: ";
        cin >> tentativa;
        tentativas++;

        if (tentativa > numero_secreto) {
            cout << "Muito alto! Tente novamente." << endl;
        } else if (tentativa < numero_secreto) {
            cout << "Muito baixo! Tente novamente." << endl;
        } else {
            cout << "Parabéns! Você acertou em " << tentativas << " tentativas." << endl;
        }
    } while (tentativa != numero_secreto);

    return 0;
}
