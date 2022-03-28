# -*- coding: utf-8 -*-
import os
#Métodos para checar o estágio em que está o sistema, para não repetir execuções e trechos já rodados anteriormente.
def check_dataframe(name, comp_nome=''):
    path = "../experimentos/%s/tabelas/"%(name)
    path_of_table = "../experimentos/%s/tabelas/%s%s.csv"%(name, name, comp_nome)
    files_in_dir = set(os.listdir(path))
    files = [path for path in files_in_dir if os.path.exists(path_of_table)]
    if files:
        print("\nDataFrame processado anteriormente!")
        return True
    else:
        return False

def check_model(name):
    #Path to tables directory.
    path = "../experimentos/%s/model/"%(name)
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    matching = [s for s in onlyfiles if "trained_model" in s]
    #print("matching:\n", matching)
    if matching:
        print("\nModelo treinado anteriormente!")
        return True
    else:
        return False

def check_quantum(name):
    #Path to tables directory.
    path = "../experimentos/%s/quantum/"%(name)
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    matching = [s for s in onlyfiles if "rslts-circs-quanticos"  in s]
    #print("matching:\n", matching)
    if matching:
        print("\nModelo treinado anteriormente!")
        return True
    else:
        return False