# -*- coding: utf-8 -*-
import os


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def make_dir(name):
    #Diretório pai
    parent_dir ="../experimentos/"
    #Diretório
    directory = str(name)
    # Path
    path = os.path.join(parent_dir, directory)
    #Se não já existir, cria novo diretório.
    if not os.path.exists(path):
        try:
            os.mkdir(path)
            print("Directory '% s' created" % directory)
        except OSError as e:
            print("Can't create {dir}: {err}".format(dir=path, err=e))


    return path