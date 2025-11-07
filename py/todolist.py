import os

ARQ = "tarefas.txt"

def carregar_tarefas():
    if not os.path.exists(ARQ):
        return []
    with open(ARQ, "r") as f:
        return [linha.strip() for linha in f.readlines()]

def salvar_tarefas(tarefas):
    with open(ARQ, "w") as f:
        for tarefa in tarefas:
            f.write(tarefa + "\n")

def listar_tarefas(tarefas):
    if not tarefas:
        print("\n Nenhuma tarefa adicionada ainda.\n")
        return
    print("\n Lista de Tarefas:")
    for i, tarefa in enumerate(tarefas, start=1):
        print(f"{i}. {tarefa}")
    print()

def adicionar_tarefa(tarefas):
    tarefa = input("Digite a nova tarefa: ")
    tarefas.append(tarefa)
    salvar_tarefas(tarefas)
    print(" Tarefa adicionada!\n")

def remover_tarefa(tarefas):
    listar_tarefas(tarefas)
    try:
        indice = int(input("Número da tarefa para remover: ")) - 1
        if 0 <= indice < len(tarefas):
            removida = tarefas.pop(indice)
            salvar_tarefas(tarefas)
            print(f" Tarefa removida: {removida}\n")
        else:
            print("Número inválido.\n")
    except ValueError:
        print("Digite um número válido.\n")

def menu():
    tarefas = carregar_tarefas()
    while True:
        print("GERENCIADOR DE TAREFAS")
        print("1. Listar tarefas")
        print("2. Adicionar tarefa")
        print("3. Remover tarefa")
        print("4. Sair")
        opc = input("Escolha: ")

        if opc == "1":
            listar_tarefas(tarefas)
        elif opc == "2":
            adicionar_tarefa(tarefas)
        elif opc == "3":
            remover_tarefa(tarefas)
        elif opc == "4":
            print("👋 Saindo... Até mais!")
            break
        else:
            print(" Opção inválida!\n")

menu()
