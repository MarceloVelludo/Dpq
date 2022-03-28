# -*- coding: utf-8 -*-
#Gera relatorio das observaveis em relação ao tempo.
def make_report_ox(self):
    #Para os pontos minimos
    dpqMin = DPQ.DinamicaPontosQuanticos(j_1_inicial=self.j_1_inicial, j_1_final=self.j_1_inicial, passoJ_1 = 0.5,
                                         j_2_inicial=self.j_2_inicial, j_2_final=self.j_2_inicial, passoJ_2 = 0.5,
                                         bz_1_inicial=self.bz_1_inicial, bz_1_final=self.bz_1_inicial, passoBz_1 = 0.5,
                                         bz_2_inicial=self.bz_2_inicial, bz_2_final=self.bz_2_inicial, passoBz_2 = 0.5,
                                         j_12_inicial=self.j_12_inicial, j_12_final=self.j_12_inicial, passoJ_12 = 0.1,
                                         tInicial=1, tFinal=20, passoT=1)
    ox1, ox2, oy1, oy2, oz1, oz2, tempos = dpqMin.criaFrameGraficos()
    dpqMin.criaGraficos(ox1, tempos, "Observável ox1- " + dpqMin.name)
    dpqMin.criaGraficos(ox2, tempos, "Observável ox2- " + dpqMin.name)
    dpqMin.criaGraficos(oy1, tempos, "Observável oy1- " + dpqMin.name)
    dpqMin.criaGraficos(oy2, tempos, "Observável oy2- " + dpqMin.name)
    dpqMin.criaGraficos(oz1, tempos, "Observável oz1- " + dpqMin.name)
    dpqMin.criaGraficos(oz2, tempos, "Observável oz2- " + dpqMin.name)

    #Para os pontos Maximos
    dpqMax = DPQ.DinamicaPontosQuanticos(j_1_inicial=self.j_1_final, j_1_final=self.j_1_final, passoJ_1 = 0.5,
                                         j_2_inicial=self.j_2_final, j_2_final=self.j_2_final, passoJ_2 = 0.5,
                                         bz_1_inicial=self.bz_1_final, bz_1_final=self.bz_1_final, passoBz_1 = 0.5,
                                         bz_2_inicial=self.bz_2_final, bz_2_final=self.bz_2_final, passoBz_2 = 0.5,
                                         j_12_inicial=self.j_12_final, j_12_final=self.j_12_final, passoJ_12 = 0.1,
                                         tInicial=1, tFinal=20, passoT=1)
    ox1, ox2, oy1, oy2, oz1, oz2, tempos = dpqMax.criaFrameGraficos()
    dpqMax.criaGraficos(ox1, tempos, "Observável ox1- " + dpqMax.name)
    dpqMax.criaGraficos(ox2, tempos, "Observável ox2- " + dpqMax.name)
    dpqMax.criaGraficos(oy1, tempos, "Observável oy1- " + dpqMax.name)
    dpqMax.criaGraficos(oy2, tempos, "Observável oy2- " + dpqMax.name)
    dpqMax.criaGraficos(oz1, tempos, "Observável oz1- " + dpqMax.name)
    dpqMax.criaGraficos(oz2, tempos, "Observável oz2- " + dpqMax.name)

    #Para os pontos médios
    dpqMean = DPQ.DinamicaPontosQuanticos(j_1_inicial=self.j_1_final/2, j_1_final=self.j_1_final/2, passoJ_1 = 0.5,
                                         j_2_inicial=self.j_2_final/2, j_2_final=self.j_2_final/2, passoJ_2 = 0.5,
                                         bz_1_inicial=self.bz_1_final/2, bz_1_final=self.bz_1_final/2, passoBz_1 = 0.5,
                                         bz_2_inicial=self.bz_2_final/2, bz_2_final=self.bz_2_final/2, passoBz_2 = 0.5,
                                         j_12_inicial=self.j_12_final/2, j_12_final=self.j_12_final/2, passoJ_12 = 0.1,
                                         tInicial=1, tFinal=20, passoT=1)
    ox1, ox2, oy1, oy2, oz1, oz2, tempos = dpqMean.criaFrameGraficos()
    dpqMean.criaGraficos(ox1, tempos, "Observável ox1- " + dpqMean.name)
    dpqMean.criaGraficos(ox2, tempos, "Observável ox2- " + dpqMean.name)
    dpqMean.criaGraficos(oy1, tempos, "Observável oy1- " + dpqMean.name)
    dpqMean.criaGraficos(oy2, tempos, "Observável oy2- " + dpqMean.name)
    dpqMean.criaGraficos(oz1, tempos, "Observável oz1- " + dpqMean.name)
    dpqMean.criaGraficos(oz2, tempos, "Observável oz2- " + dpqMean.name)

    return

def make_report(self, name, X_train, y_real, y_pred):
    y_real = [y_real]
    relative_error_1 = 0
    relative_error_2 = 0
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    r2 = r2_score(y_real,y_pred)

    for y_p, y_t in zip(y_pred, y_real):
        relative_error_1 += np.abs(1-y_p)/y_t
        relative_error_2 += np.abs((y_t-y_p)/y_t)

    #Font
    self.pdf.set_font("Times", 'B', 15)
    #Title
    self.pdf.cell(15,15, name,1,0, 'L')
    #Line break
    self.pdf.ln(2)
    self.pdf.set_font("Times", 'B', 13)

    #String com os resultados
    rst_string = ''.join([name, "\nMédia do erro absoluto: %f \nMédia quadrada do erro: %f \nR2: %f\nrelative_error_1(np.abs(1-y_p)/y_t): %f \nrelative_error_2(np.abs((y_t-y_p)/y_t)): %f\n" % (mae, mse, r2, relative_error_1, relative_error_2)])       
    self.pdf.cell(20,15,rst_string,1)
    print("\nRESULTADOS:", rst_string)
    #Graficos
    plotGraph(y_real, y_pred, "Extra Trees Regressor "+name,"teste", mae=mae, mse=mse, r2=r2)
    gen_acc_measures(pd.DataFrame(X_train),y_real, "teste")

    return [mae, mse, r2]

def make_speed(self,t0,t1, subname = "", name = ""):
    if name == "":
        name = self.name_comp
    with open("Speed.txt", "a+") as text_file:
        text_file.write("===="*5)
        text_file.write(subname+"\n")
        text_file.write("\n%s"%(name))
        text_file.write("Tempo: %f \n" % (t1-t0))
    return

def test_aleatorio(self, k):
    arrayJ_1 = self.random_samples(self.j_1_inicial, self.j_2_final, 10)
    arrayJ_2 = self.random_samples(self.j_2_inicial, self.j_2_final, 10)
    arrayBz_1 = self.random_samples(self.bz_1_inicial, self.bz_1_final, 10)
    arrayBz_2 = self.random_samples(self.bz_2_inicial, self.bz_2_final, 10)
    arrayJ_12 = self.random_samples(self.j_12_inicial, self.j_12_final, 10)

    self.elementos_iter = random.sample(list(product(arrayJ_1, arrayJ_2, arrayBz_1, arrayBz_2,  arrayJ_12)), k)
    #print("\n len teste aleatorio:", len(self.elementos_iter))
    df = self.criaDataFrame("teste-k-")
    X_val = df.iloc[:,5:]
    y_val = df.iloc[:,4]
    y_val_pred = self.model.predict(X_val)
    #print("len y_val_pred:",len(y_val_pred))

    self.make_report("teste aleatorio com K: %f "%k, y_val, y_val_pred)

    return

##-----------------
def make_results(self, k = 10000):
    #Hyperparametros para testes:
    #Controla o numero de arvore de decisões que vão ser usadas.
    #Muitas àrvores pode gerar overfitting, mas isso é dificil de ocorrer com extra-trees
    #n_trees = [10, 50, 100, 500, 1000, 5000]
    #O numero maximo de feature que pode ser considerado para a repartição de cada nó, parametro mais importante do modelo.
    #max_features= [1,2,3,4,5,6,7,8,9,10,11,12]
    t0 = perf_counter()
    #Se existir da load e retorna falso, caso contrário um novo frame será criado.
    if(self.check_tabela()):
        self.criaDataFrame()

    model, X_train, X_test, y_train, y_test = self.set_model()

    y_train_pred = model.predict(X_train)
    self.make_report("Treino", y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    self.make_report("Teste", y_test, y_test_pred)
    t1 = perf_counter()

    self.make_speed(subname = "Criação das tabelas e treinamento do modelo", t0 = t0, t1 = t1)

    self.saveDataFrame()
    self.save_Y(y_test, y_test_pred, len(y_test))

    t0 = perf_counter()
    self.test_aleatorio(k)
    t1 = perf_counter()
    self.make_speed(subname = "teste aleatorio",t0 = t0, t1 = t1)

    t0 = perf_counter()
    self.make_report_ox()
    t1 = perf_counter()
    self.make_speed(subname = "Relatório observáveis:",t0 = t0, t1 = t1)

    t0 = perf_counter()
    self.graph_quantumvsclassic()
    t1 = perf_counter()
    self.make_speed(subname = "Parte quântica:",t0 = t0, t1 = t1)

    return