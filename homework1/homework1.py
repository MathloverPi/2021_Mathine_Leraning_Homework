# -*- coding: utf-8 -*-
import pandas as pd

class Samples(object):
    """
    读入样本文件，划分属性
    """
    attributes = ()
    factors = {}

    def __init__(self, filename):
        """
        读入文件
        """
        self.data = pd.read_csv(filename)
        self.attributes = self.data.columns.values.tolist()[:-1]
        for i in self.attributes:
            self.factors[i] = []
            temp = list(self.data[i].drop_duplicates())
            self.factors[i] = temp
        self.data = tuple(self.data.values)

class FindS(object):
    def __init__(self, samples):

        self.num_attr = len(samples.attributes)
        self.factors = samples.factors
        self.attr = samples.attributes
        self.data_set = samples.data

    def run(self):
        """
        运行算法，随着样本的输入，更改S，最后输出S
                """
        s = self.initial_s()
        times = 0
        for sample in self.data_set:
            print('S' + str(times), s)
            times += 1
            print('第' + str(times) + '个样本', sample)
            if sample[-1]=='Yes':
                for i in range(4):
                    if(not (sample[i] in s[i])):
                        s[i].append(sample[i])
        print("S:", s)

    def initial_s(self):
        """
        初始化S，最特殊成员，每个属性对应都为空列表
        """
        return [[],[],[],[]]

class CandidateElimination(object):
    def __init__(self, samples):
        """
        初始化候选消除算法中的数据，基于类Samples
        """
        self.num_attr = len(samples.attributes)
        self.factors = samples.factors
        self.attr = samples.attributes
        self.data_set = samples.data

    def run(self):
        """
        运行算法，随着样本的输入，更改S,G，最后输出S，G，之间的所有概念
        """
        s = self.initial_s()
        g = self.initial_g()
        times = 0
        for sample in self.data_set:
            print('S' + str(times), s)
            print('G' + str(times), g)
            times += 1
            print('第' + str(times) + '个样本', sample)
            if g == [] or s == [None]:
                s = []
                break;
            if sample[-1]=='Yes':
                # 如果是正例,从G中移去所有与样本不一致的假设,将S变为与样例一致的最特殊的假设；
                g = self.remove_inconsistent_g(g, sample[:-1])
                s_new = s[:]
                for each_s in s:
                    if not self.consistent(each_s, sample):
                        s_new.remove(each_s)
                        s_mini_paradigm = self.minimalist_paradigm(each_s, sample, g)
                        s_new.append(s_mini_paradigm)
                        s_new = self.remove_more_general(s_new)
                s = s_new[:]
            elif sample[-1]=='No':
                # 如果是反例,从S中移去所有与样本不一致的假设,将G变为与样例一致的最一般的假设；
                s = self.remove_inconsistent_s(s, sample[:-1])
                g_new = g[:]
                for each_g in g:
                    if self.consistent(each_g, sample):
                        g_new.remove(each_g)
                        g_mini_special = self.minimal_specialization(each_g, sample, s)
                        g_new += g_mini_special
                        g_new = self.remove_more_special(g_new)
                g = g_new[:]
        print("S:", s)
        print("G:", g)
        self.get_concept(s, g)

    def initial_s(self):
        """
        初始化S，最特殊成员，每个属性对应都为空，用/代替
        """
        return [tuple(['/' for factor in range(self.num_attr)])]

    def initial_g(self):
        """
        初始化G，最一般成员，每个属性可取任何值，用?代替
        """
        return [tuple(['?' for factor in range(self.num_attr)])]

    def remove_inconsistent_g(self, g, sample):
        """
        从G中移去所有与训练样例不一致的假设
        """
        set_new = g[:]
        for each_set in set_new:
            if not self.consistent(each_set, sample):
                g.remove(each_set)
        return g

    def remove_inconsistent_s(self, s, sample):
        """
        从S中移去所有与训练样例不一致的假设
        """
        set_new = s
        for each_set in set_new:
            if self.consistent(each_set, sample):
                set_new.remove(each_set)
        return set_new

    def consistent(self, a, b):
        """
        判断两个概念是否一致
        """
        for i in range(self.num_attr):
            if not self.match_factor(a[i], b[i]):
                return False
        return True

    @staticmethod
    def match_factor(i, j):
        """
        判断两个概念中对应的属性是否一致
        """
        if i == '?' or j == '?':
            return True
        elif i == j:
            return True
        return False

    def minimalist_paradigm(self, concept, sample, g):
        """
        把S中概念的所有的极小泛化式h加入到S中，其中h满足与d一致，而且G的某个成员比h更一般
        """
        hypo = list(concept)
        for i, factor in enumerate(hypo):
            if factor == '/':
                hypo[i] = sample[i]
            elif not self.match_factor(factor, sample[i]):
                hypo[i] = '?'
        h = tuple(hypo)
        for each_g in g:
            if self.more_general(each_g, h):
                return h
        return None

    @staticmethod
    def more_general(a, b):
        """
        判断a是否比b更一般
        """
        hyp = zip(a, b)
        for i, j in hyp:
            if i == '?':
                continue
            elif j == '?':
                if i != '?':
                    return False
            elif i != j:
                return False
            else:
                continue
        return True

    def remove_more_general(self, s):
        """
        移除S更加中更为一般的例子
        """
        for s_i in s:
            for s_j in s:
                if s_i != s_j and self.more_general(s_i, s_j):
                    s.remove(s_j)
        return list(set(s))

    def minimal_specialization(self, concept, sample, s):
        """
        把G中概念的所有的极小泛化式h加入到G中，其中h满足与d一致，而且S的某个成员比h更特殊
        """
        h = []
        hypo = list(concept)
        for i, factor in enumerate(hypo):
            if factor == '?':
                values = self.factors[self.attr[i]]
                for j in values:
                    if sample[i] != j:
                        hyp = hypo[:]
                        hyp[i] = j
                        for k, each in enumerate(hyp):
                            if each == "?":
                                continue
                            elif each == sample[k]:
                                hyp[k] = '?'
                        hyp = tuple(hyp)
                        for each_s in s:
                            if self.more_general(hyp, each_s) or each_s == self.initial_s()[0]:
                                h.append(hyp)
                                break
        return h

    def remove_more_special(self, g):
        """
        从G中移除更特殊的概念
        """
        for g_i in g:
            for g_j in g:
                if g_i != g_j and self.more_general(g_j, g_i):
                    g.remove(g_j)
        return list(set(g))

    def get_concept(self, s, g):
        """
        得到S和G中间的概念
        """
        concepts = []
        for each_s in s:
            for each_g in g:
                for i in range(self.num_attr):
                    new_concept = list(each_g)[:]
                    if each_s[i] == each_g[i]:
                        continue
                    elif each_g[i] == '?':
                        new_concept[i] = each_s[i]
                        concepts.append(tuple(new_concept))
        print(set(self.remove_more_special(concepts)))


if __name__ == "__main__":
    samples = Samples("samples.csv")
    myFindS = FindS(samples)
    myFindS.run()
    print('='*100)
    myCandidateElimination = CandidateElimination(samples)
    myCandidateElimination.run()