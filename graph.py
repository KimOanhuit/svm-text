import snap
import sys
import math
import random
import time

class Graph:
    # negative = open('NegativeDataset.txt', 'r')
    # positive = open('PositiveDataset.txt', 'r')
    # base = open('datatest.txt', 'r')
    base = open('DatasetFilter.txt', 'r')

    def __init__(self):

        self.graph = snap.TNGraph.New()

    def filterNeutralFromDataset(self):
        f = open('test.txt', 'a+')
        data = ''
        check = 0

        for line in sys.stdin.readlines():
            data += line

            if line == 'VOT:0\n':
                check = 1    
            if line == '\n':
                if check == 0:
                    f.write(data)
                check = 0
                data = ''
        
    def createGraph(self):
        find_user = dict()
        information = dict() #'SRC_ID TGT_ID':'VOT','RES','YEA','DAT','TXT'
        left_info = []
        right_info = []
        id = 0

        for line in self.base.readlines():
            if line == '\n':
                continue
            line = line.replace("\n","")
            part = line.split(":")
            if part[0] == "SRC":
                if part[1] not in find_user:
                    id += 1
                    find_user[part[1]] = id
                    self.graph.AddNode(id)
                left_info.append(find_user[part[1]])
            elif part[0] == "TGT":
                if part[1] not in find_user:
                    id += 1
                    find_user[part[1]] = id
                    self.graph.AddNode(id)
                left_info.append(find_user[part[1]])
                self.graph.AddEdge(left_info[0], left_info[1])
            elif part[0] == "TXT":
                right_info.append(part[1])
                information["".join(str(left_info))] = right_info
                right_info = []
                left_info = []
            else:
                right_info.append(part[1])

        # snap.SaveEdgeList(self.graph, "GraphWiki.txt", "List of edges")
        # print "graph: Nodes %d, Edges %d" % (self.graph.GetNodes(), self.graph.GetEdges())
        return information
        
    
    def balanceGraph(self):
        dict = {}
        d = self.createGraph()

        print "get negative node"
        count = 0
        for key in d.keys():
            if d[key][0] == '-1':
                count += 1
                dict[key] = d[key]
                del d[key]
        print "get complete"

        print "get positive node"
        print "count: " + str(count)
        # Random canh duong
        for i in range(1,count+1):
            key = random.choice(d.keys())
            dict[key] = d[key]
            del d[key]
            print "Complete: " + str(i) + "/" + str(count)
        print "get complete"

        balanceGraph = Graph()
        balanceGraph.convertDictionaryToGraph(dict)

        return balanceGraph

    def convertDictionaryToGraph(self, dict):
        l = []
        for key in dict.keys():
            input = key.replace('[', '').replace(']', '').replace(',', '')
            find = input.find(' ')
            fromNode = int(input[:find])
            toNode = int(input[find:])

            if fromNode not in l:
                l.append(fromNode)
                self.graph.AddNode(fromNode)
            if toNode not in l:
                l.append(toNode)
                self.graph.AddNode(toNode)
                
        
            self.graph.AddEdge(fromNode, toNode)
        
        snap.SaveEdgeList(self.graph, "GraphWiki_Balance.txt", "List of edges")

    # def subGraph(self):
        



            

        

        


            
        





           











