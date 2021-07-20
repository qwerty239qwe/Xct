colorSpace <- c('#E41A1C','#377EB8','#4DAF4A','#984EA3','#F29403','#F781BF','#BC9DCC','#A65628','#54B0E4','#222F75','#1B9E77','#B2DF8A',
                '#E3BE00','#FB9A99','#E7298A','#910241','#00CDD1','#A6CEE3','#CE1261','#5E4FA2','#8CA77B','#00441B','#DEDC00','#B3DE69','#8DD3C7','#999999')

library(igraph)
library(grDevices)
CC <- read.csv(file = './CCC_HumanSkin.csv', row.names = 'X', stringsAsFactors = FALSE)
CC[CC<0.2] <- 0 #filter
CC <- as.matrix(CC)


network <- graph_from_adjacency_matrix(CC, mode = 'directed', weighted = T)
#V(network)
#vertex_attr(network)
#E(network)
#edge_attr(network)

# plot it
color.use <- colorSpace[1:dim(CC)[1]]
V(network)$color <- color.use
V(network)$frame.color <- color.use
V(network)$size<- 20
V(network)$label.color <- 'black'
V(network)$label.cex <- 1.2 #font

tails <- as.vector(tail_of(network, E(network)))
E(network)$color <- V(network)$color[tails] #edge colors follow vertices
E(network)$color <- adjustcolor(E(network)$color, alpha.f = 0.6)
E(network)$width <- 3*E(network)$weight
E(network)$arrow.size <- 0.2

png('CCC.png', width = 4000, height = 4000, res = 300)
plot(network, edge.curved=0.3, vertex.label.dist = 0.1,
     layout=layout_in_circle, margin = 0.1, vertex.label.family="Times")
dev.off()


#toy example
inc <- matrix(sample(0:2, 16, repl=TRUE), 4, 4)
colnames(inc) <- LETTERS[1:4]
rownames(inc) <- LETTERS[1:4]
toynet <- graph_from_adjacency_matrix(inc, mode = 'directed', weighted = T)
plot(toynet, edge.width=abs(edge_attr(toynet)$weight)*3)



