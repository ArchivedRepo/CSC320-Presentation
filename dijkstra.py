from node import Node
import heapq

def graph_search(G, start, end):
    """
    Conduct a graph sarch and return a chain of 
    """
    current_node = Node(start[0], start[1])
    visited = set()
    node_dict = {}
    node_dict[start] = current_node
    q = [(0, start)]
    # in_q_node store tuple
    in_q_nodes = set()
    in_q_nodes.add(start)
    while True:
        cost, node_pos = heapq.heappop(q)
        node_obj = node_dict[node_pos]
        visited.add(node_pos)
        if node_pos == end:
            return node_obj
        for row, col in G[node_pos]:
            if (row, col) not in visited:
                g_tmp = node_obj.cost + G[node_pos][(row, col)]
                if (row, col) in in_q_nodes and node_dict[(row,col)].cost > g_tmp:
                    q.remove((node_dict[(row,col)].cost, (row,col)))
                    in_q_nodes.remove((row, col))
                    heapq.heapify(q)
                if (row, col) not in in_q_nodes:
                    node_dict[(row,col)] = Node(row, col)
                    node_dict[(row,col)].cost = g_tmp
                    node_dict[(row,col)].pred = node_obj
                    heapq.heappush(q, (g_tmp, (row, col)))
                    in_q_nodes.add((row, col))

