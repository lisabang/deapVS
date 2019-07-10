import copy as cp
import functools
import itertools as itert
import multiprocessing
import random
import numpy as np
from deap import creator, base, tools, algorithms
from numpy.random import RandomState
from .mlr_eval import mlr_r2


def hash_ind_list(i):
    return hash(tuple(i))


# arguments:  ngen, basetable, y, popsize, indsize, crossoverrate, #mutprob, evaluation function, selection function
toolbox = base.Toolbox()


class GAdescsel:
    def __init__(
        self,
        basetable,
        y,
        ngen=1000,
        popsize=100,
        indsize=5,
        cx=0.5,
        mut=0.05,
        seed=int(12345),
    ):
        """
        initialize GA object, giving ngen, popsize, indsize, 
        crossover rate, mutation rate, and random seed initialization
        """
        creator.create("Fitness", base.Fitness, weights=(1.0,))

        creator.create(
            "Individual", list, fitness=creator.Fitness, __hash__=hash_ind_list
        )
        creator.create(
            "Population", list, fitness=creator.Fitness, __hash__=hash_ind_list
        )
        # toolbox=base.Toolbox()
        global toolbox
        # global evalq2loo
        self.seed = seed
        self.basetable = basetable
        self.y = y
        self.ngen = ngen
        self.popsize = popsize
        self.indsize = indsize
        self.cx = cx
        self.mut = mut
        pool = multiprocessing.Pool()

    def ct_calls(func):
        """
        this is a decorator function to count the number of calls to self.mkeinseed.count so that we can make the
        the random number generator use a different seed each time (increases by one each time)
        :return: number of times mkeinseed has been called
        """

        @functools.wraps(func)
        def decor(*args, **kwargs):
            decor.count += 1
            return func(*args, **kwargs)

        decor.count = 0
        return decor

    def mkeindrand(self, desc_in_ind=5):
        """
        :param desc_in_ind: number of descriptors in model ("individual" in deap)
        :return: a random sample
        """
        while str(type(self.basetable)) != "<class 'pandas.core.frame.DataFrame'>":
            raise TypeError(
                "The type of descriptor table should be a Pandas dataframe."
            )
        while type(desc_in_ind) is not int:
            try:
                print
                "converting non-int to int"
                desc_in_ind = int(desc_in_ind)
                break
            except:
                raise ValueError(
                    "The number of descriptors per individual should be of type int"
                )
        print(random.sample(set(self.basetable.columns), 5))
        smple = random.sample(set(self.basetable.columns), desc_in_ind)

        return smple

    @ct_calls
    def mkeindseed(self, desc_in_ind=5):
        if self.mkeindseed.count <= 100:
            prng = RandomState(self.seed + int(self.mkeindseed.count))
        if self.mkeindseed.count > 100:
            prng = RandomState(self.seed + int((self.mkeindseed.count % 100)))
        smple = prng.choice(self.basetable.columns, size=desc_in_ind, replace=False)
        return list(smple)

    def mutaRan(self, ind):
        # mutpool=[str(i) for i in ndesc.index if i not in ind]
        for descriptor in ind:
            if np.random.binomial(1, self.mut, 1) == 1:
                choices = [x for x in list(self.basetable.columns) if x not in ind]
                ind[ind.index(descriptor)] = random.choice(choices)
        return (ind,)

    def evalr2(self, ind):
        return (mlr_r2(self.basetable[ind], self.y)[0],)

    def evalr2adj(self, ind):
        return (mlr_r2(self.basetable[ind], self.y)[1],)

    def evalq2loo(self, ind):
        #        print self.basetable[ind][1]
        return (q2loo_mlr(self.basetable[ind], self.y),)

    def printq2fitness(self, pop):
        # this needs rewriting
        q2s = []
        for ind in pop:
            q2s.append(IQSAR.mlr3.q2loo_mlr(self.basetable[ind], self.y))
        return q2s

    def pretty_print(self, evolveobj):
        """
        pretty prints the result of evolve function to pd dataframe
        """

        origdf = pd.DataFrame.from_records(evolveobj[0])
        origdf["scores"] = pd.DataFrame.from_records(list(evolveobj[1]))
        finaldf = pd.DataFrame.from_records(evolveobj[2])
        finaldf["scores"] = pd.DataFrame.from_records(list(evolveobj[3]))
        self.evo_o = origdf
        self.evo_f = finaldf
        return origdf, finaldf

    def evolve(self, evalfunc="q2loo"):
        """
        1st element returned is the original population, 
        2nd is is the evaluation of the firness function on the originalpopualtion, 
        3rd is the final population, 
        4th is the evalution of the fitness function on the final population.
 
        """
        toolbox.register("genind", self.mkeindseed, self.indsize)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.genind)
        toolbox.register("population",tools.initRepeat, list, toolbox.individual, n=self.popsize)
        
        if evalfunc=="q2loo":
            toolbox.register("evaluate", self.evalq2loo)
        elif evalfunc=="q2lmo":
            toolbox.register("evaluate", self.evalq2lmo)
        elif evalfunc=="r2":
            toolbox.register("evaluate", self.evalr2)
        elif evalfunc=="r2adj":
            toolbox.register("evaluate", self.evalr2adj)
        else:
            raise ValueError("not a valid evaluation function specified; use evalr2adj, evalr2, or q2loo")
        
        toolbox.register("mate", tools.cxOnePoint) #Uniform, indpb=0.5)
        toolbox.register("mutate", self.mutaRan)#, indpb=self.mut)
        toolbox.register("select", tools.selBest)
        #progress bar start!
        #print 'Starting... # GEN FINISHED:',

        origpop=toolbox.population()
        #self.mkeindseed.count=0
        population=cp.deepcopy(origpop)
        fits=toolbox.map(toolbox.evaluate, population)
        for fit, ind in zip(fits,population):
            ind.fitness.values=fit
        
        avgfitnesses=[]
        popfits=0
        #prb=ProgressBar(self.ngen)
        for gen in range(self.ngen):
            try:
                offspring=algorithms.varOr(population, toolbox, lambda_=self.popsize, cxpb=self.cx, mutpb=self.mut)   
                for ind in offspring:
                    ind.fitness.values=toolbox.evaluate(ind)
                population=toolbox.select([k for k,v in itert.groupby(sorted(offspring+population))], k=100)
                popfits = toolbox.map(toolbox.evaluate, population)
                #prb.animate(gen)
                #prb.score=np.mean(popfits)
                #ProgressBar.score=property(lambda self: self.score+np.mean(popfits))
                #prb.update_time(1, prb.score)
            except (KeyboardInterrupt, SystemExit):
                result = [origpop, toolbox.map(toolbox.evaluate, origpop), population, toolbox.map(toolbox.evaluate, population)]
                return result#self.pretty_print(returnobj)
        result = [origpop, toolbox.map(toolbox.evaluate, origpop), population, toolbox.map(toolbox.evaluate, population)]
        return self.pretty_print(result)
    def get_df(self, chosenind):
        btt = self.basetable[chosenind]

        print(
            "r2 is: ",
            mlr.mlr(btt, self.y)[2],
            "r2adj is: ",
            mlr.mlr(btt, self.y)[3],
            "q2loo is: ",
            mlr.q2loo_mlr(btt, self.y),
        )
        print("coefficients are:", m.mlr(btt, self.y)[0])
        return btt

    def debug_eval(self):
        toolbox.register("evaluate", evalr2, self.y, self.basetable)
        toolbox.register("mate", tools.cxOnePoint)  # Uniform, indpb=0.5)
        toolbox.register("mutate", mutRan, indpb=self.mut)
        toolbox.register("select", tools.selBest)
        population = toolbox.population()
        fits = toolbox.map(toolbox.evaluate, population)

        for fit, ind in zip(fits, population):
            ind.fitness.values = fit
        offspring = algorithms.varOr(
            population, toolbox, lambda_=100, cxpb=0.5, mutpb=0.05
        )
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
            print(ind)
            print(ind.fitness.values)
