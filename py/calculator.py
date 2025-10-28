#Calculadora simples em py

def sum(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "Erro: divisão por zero!"
    return a / b

#menu
def main():
    print("Calculadora simples em Python")
    print("Escolha a operação:")
    print("1 - Soma")
    print("2 - Subtração")
    print("3 - Multiplicacao")
    print("4 - Divisão")

    escolha = input("Digite sua escolha (1/2/3/4): ")

    a = float(input("Digite o primeiro número: "))
    b = float(input("Digite o segundo número: "))

    if escolha == '1':
        print("Resultado:", sum(a, b))
    elif escolha == '2':
        print("Resultado:", subtract(a, b))
    elif escolha == '3':
        print("Resultado:", multiply(a, b))
    elif escolha == '4':
        print("Resultado:", divide(a, b))
    else:
        print("Opção inválida.")

if __name__ == "__main__":
    main()
