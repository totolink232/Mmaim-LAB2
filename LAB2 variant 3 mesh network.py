import os
import heapq
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

base_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(base_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# -------------------------------------------------------------
# Варіант 3: Комп'ютерна мережа з 9 вузлів, топологія "повна сітка" (mesh)
# -------------------------------------------------------------
class MeshNetwork:
    def __init__(self, n_nodes=9):
        self.n = n_nodes
        self.node_names = [f"N{i+1}" for i in range(n_nodes)]
        self.name_to_id = {name: i for i, name in enumerate(self.node_names)}
        
        # матриця суміжності та список суміжності
        self.adj_matrix = [[0] * n_nodes for _ in range(n_nodes)]
        self.adj_list = {i: [] for i in range(n_nodes)}
        self.edges = []
        
        # генеруємо симетричні ваги затримок для каналів (від 2 до 15 мс)
        np.random.seed(42)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                w = int(np.random.randint(2, 16))
                if (i == 0 and j == 8) or (i == 8 and j == 0):
                    w = 18
                
                self.adj_matrix[i][j] = w
                self.adj_matrix[j][i] = w
                self.adj_list[i].append((j, w))
                self.adj_list[j].append((i, w))
                self.edges.append((self.node_names[i], self.node_names[j], w))

    def get_characteristics(self):
        degrees = [len(self.adj_list[i]) for i in range(self.n)]
        total_edges = len(self.edges)
        handshake_sum = sum(degrees)
        unweighted_diam = 1
        
        all_dists = []
        for i in range(self.n):
            dist, _ = self.dijkstra(i)
            all_dists.extend(dist)
        weighted_diam = max(all_dists)
        
        return {
            'nodes_count': self.n,
            'edges_count': total_edges,
            'degrees': degrees,
            'handshake_sum': handshake_sum,
            'unweighted_diameter': unweighted_diam,
            'weighted_diameter': weighted_diam
        }

    def dijkstra(self, start_idx, target_idx=None):
        dist = [float('inf')] * self.n
        parent = [-1] * self.n
        dist[start_idx] = 0
        pq = [(0, start_idx)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if target_idx is not None and u == target_idx:
                break

            for v, weight in self.adj_list[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    parent[v] = u
                    heapq.heappush(pq, (dist[v], v))

        path = []
        if target_idx is not None:
            curr = target_idx
            while curr != -1:
                path.append(curr)
                curr = parent[curr]
            path.reverse()

        return dist, path

def run_variant3():
    print("=" * 70)
    print("ІНДИВІДУАЛЬНЕ ЗАВДАННЯ: ВАРІАНТ № 3")
    print("Мережа з 9 вузлів, топологія 'повна сітка' (mesh / complete graph K9)")
    print("=" * 70)

    mesh = MeshNetwork(9)
    char = mesh.get_characteristics()

    print(f"Кількість вузлів (вершин):   {char['nodes_count']}")
    print(f"Кількість каналів (ребер):   {char['edges_count']} (за формулою n*(n-1)/2 = 9*8/2 = 36)")
    print(f"Степінь кожної вершини:      deg(v) = {char['degrees'][0]}")
    print(f"Сума степенів:               {char['handshake_sum']} (2 * |E| = {2 * char['edges_count']}, закон сходиться)")
    print(f"Діаметр графа (кількість хопів): {char['unweighted_diameter']} хоп")
    print(f"Зважений діаметр мережі:     {char['weighted_diameter']} мс")

    # Знаходження найкоротшого маршруту між N1 та N9
    dist, path = mesh.dijkstra(0, 8)
    path_names = [mesh.node_names[idx] for idx in path]
    print(f"\nНайкоротший маршрут між N1 та N9: {' -> '.join(path_names)}")
    print(f"Сумарна затримка: {dist[8]} мс (прямий лінк мав би затримку 18 мс)")

    G = nx.Graph()
    for name in mesh.node_names:
        G.add_node(name)
    for u, v, w in mesh.edges:
        G.add_edge(u, v, weight=w)

    pos = nx.circular_layout(G)
    plt.figure(figsize=(9, 8))
    nx.draw_networkx_edges(G, pos, width=1.2, edge_color='#888888', alpha=0.7)
    nx.draw_networkx_nodes(G, pos, node_color='#1f77b4', node_size=1100)
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=11, font_weight='bold')
    
    plt.title("Топологія комп'ютерної мережі 'Повна сітка' (Full Mesh, 9 вузлів, 36 каналів)", fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'variant3_full_mesh_topology.png'), dpi=300)
    plt.close()
    print("Графік топології збережено в plots/variant3_full_mesh_topology.png")

    plt.figure(figsize=(9, 8))
    nx.draw_networkx_edges(G, pos, width=1.0, edge_color='#cccccc', alpha=0.5)
    
    path_edges = list(zip(path_names[:-1], path_names[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3.5, edge_color='#d9534f')
    
    node_colors = []
    for node in G.nodes():
        if node == 'N1':
            node_colors.append('#5cb85c')
        elif node == 'N9':
            node_colors.append('#d9534f')
        elif node in path_names:
            node_colors.append('#f0ad4e')
        else:
            node_colors.append('#6c757d')
            
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1100)
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=11, font_weight='bold')

    edge_labels = {edge: f"{G.edges[edge]['weight']} мс" for edge in path_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='#d9534f')

    plt.title(f"Найкоротший маршрут Дейкстри від N1 до N9 ({' -> '.join(path_names)}, затримка {dist[8]} мс)", fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'variant3_shortest_path_highlighted.png'), dpi=300)
    plt.close()
    print("Графік найкоротшого шляху збережено в plots/variant3_shortest_path_highlighted.png")

if __name__ == '__main__':
    run_variant3()
