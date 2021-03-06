import pygraphviz as PG

A = PG.AGraph(directed = True, strict = True)

A.add_edge("7th Edition", "32V")
A.add_edge("7th Edition", "Xenix")
# etc., etc.

# save the graph in dot format
A.write('ademo.dot')

# pygraphviz renders graphs in neato by
default,
# so you need to specify dot as the layout engine
A.layout(prog = 'dot')

# opening the dot file in a text editor shows the graph 's syntax:
digraph unix {
   size = "7,5";
   node[color = goldenrod2, style = filled];
   "7th Edition" - > "32V";
   "7th Edition" - > "V7M";
   "7th Edition" - > "Xenix";
   "7th Edition" - > "UniPlus+";
   "V7M" - > "Ultrix-11";
   "8th Edition" - > "9th Edition";
   "1 BSD" - > "2 BSD";
   "2 BSD" - > "2.8 BSD";
   "2.8 BSD" - > "Ultrix-11";
   "2.8 BSD" - > "2.9 BSD";
   "32V" - > "3 BSD";
   "3 BSD" - > "4 BSD";
   "4 BSD" - > "4.1 BSD";
   "4.1 BSD" - > "4.2 BSD";
   "4.1 BSD" - > "2.8 BSD";
   "4.1 BSD" - > "8th Edition";
   "4.2 BSD" - > "4.3 BSD";
   "4.2 BSD" - > "Ultrix-32";
}
