# from graph import Graph
import snap
import sys
import matplotlib.pyplot

# g = Graph()

# g.filterNeutralFromDataset()
# print "Create Graph:"
# g.createGraph()
# print "Complete Create Graph:"
# print "Create Balance Graph:"
# g.balanceGraph()
# print "Complete Create Balance Graph:"

G = snap.LoadEdgeList(snap.PNGraph, "GraphWiki_Balance.txt", 0, 1)
snap.PlotInDegDistr(G, "wikiPlot", "wiki-rfa")