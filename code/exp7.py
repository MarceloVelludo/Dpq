from eq_classica import Eq_cl
from quantum import Qt
from model import Md
from time import perf_counter
import numpy as np

def exp_full():
    mhz = 10**(-3)
    pi = np.pi
    j_1 = 280*mhz*2*pi
    j_2 = 320*mhz*2*pi
    bz_1 = (pi/16)*(10**3)*mhz
    bz_2 = (pi/16)*(10**3)*mhz
    j_12 = pi/140
    temp_0 = perf_counter()
    print("\n0-Inicializando modelo classico...")
    classica = Eq_cl(j_1_inicial= j_1*0.8, j_1_final= j_1*1.2, passoJ_1 = 0.0001,
                                      j_2_inicial= j_2*0.8, j_2_final= j_2*1.2, passoJ_2 = 0.0001,
                                      bz_1_inicial= bz_1*0.8, bz_1_final= bz_1*1.2, passoBz_1 = 0.1,
                                      bz_2_inicial= bz_2*0.8, bz_2_final= bz_2*1.2, passoBz_2 = 0.1,
                                      j_12_inicial= j_12*0.8, j_12_final= j_12*1.2, passoJ_12 = 0.02,
                                      tInicial=1, tFinal=20, passoT=1)

    temp_1 = perf_counter()
    print("\nInicializado com sucesso Data frame criado com sucesso!")
    print("Tempo:", temp_1-temp_0)

    print("\n1-Criando data frame...")
    classica.criaDataFrame()
    temp_2 = perf_counter()
    print("\nData frame criado com sucesso!")
    print("Tempo:", temp_2-temp_1)
    print("\n1.2-Criando Graficos das Observaveis...")
    classica.make_report_ox()
    temp_3 = perf_counter()
    print("\nData frame criado com sucesso!")
    print("Tempo:", temp_3-temp_2)

    print("2-Construindo modelo...")
    model = Md(df = classica.dataset, name=classica.name, k=10000, pdf=classica.pdf )
    model.set_model()
    temp_4 = perf_counter()
    print("Modelo construido com sucesso!")
    print("Tempo:", temp_4-temp_3)

    model.model_results(df_ale = classica.test_aleatorio())
    temp_5 = perf_counter()
    print("Predições realizadas!")
    print("Tempo:", temp_5-temp_4)

    print("3-Construindo simulação quântica...")
    results_min, results_max, results_median = classica.sample_min_max_median(n= 0)
    quantum = Qt(classica.name, model = model, pdf = model.pdf)
    quantum.elements_to_qt(results_min, results_max, results_median)
    temp_6 = perf_counter()
    print("Simulação construida com sucesso!")
    print("Tempo:", temp_6-temp_5)
    print("Tempo Total:%s"%(str((temp_6-temp_0)/60))+" minutos")
    quantum.pdf.write(5,"Tempo ini. modelo clássico:%s\nTempo modelo clássico data frame:%s\nTempo gráfico obs:%s\nTempo modelo ml:%s\nTempo predições:%s\nTempo Simulação quântica:%s\nTempo Total:%s\n"%((temp_1-temp_0),(temp_2-temp_1),(temp_3-temp_2),(temp_4-temp_3),(temp_5-temp_4),(temp_6-temp_5),(temp_6-temp_0)))
    quantum.output()
    return
exp_full()