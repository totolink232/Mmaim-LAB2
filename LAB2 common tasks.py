import os
import heapq
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx

base_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(base_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# -------------------------------------------------------------
# Клас Graph із Завдання 4 (розширений методами BFS, DFS, циклів)
# -------------------------------------------------------------
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.adj_matrix = [[0] * vertices for _ in range(vertices)]
        self.adj_list = {i: [] for i in range(vertices)}

    def add_edge(self, u, v, weight=1, directed=False):
        self.adj_matrix[u][v] = weight
        self.adj_list[u].append((v, weight))
        if not directed:
            self.adj_matrix[v][u] = weight
            self.adj_list[v].append((u, weight))

    def print_adj_matrix(self):
        for row in self.adj_matrix:
            print(row)

    def print_adj_list(self):
        for vertex, neighbors in self.adj_list.items():
            print(f"{vertex}: {neighbors}")

    def degree(self, v):
        return len(self.adj_list[v])

    def is_connected(self):
        # перевірка зв'язності через BFS
        if self.V == 0:
            return True
        visited = [False] * self.V
        queue = deque([0])
        visited[0] = True
        count = 1

        while queue:
            u = queue.popleft()
            for v, _ in self.adj_list[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    count += 1
        return count == self.V

    def dijkstra(self, start):
        # алгоритм Дейкстри для найкоротших відстаней
        dist = [float('inf')] * self.V
        dist[start] = 0
        pq = [(0, start)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            for v, weight in self.adj_list[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    heapq.heappush(pq, (dist[v], v))
        return dist

    def bfs(self, start):
        # пошук у ширину (BFS)
        visited = [False] * self.V
        order = []
        queue = deque([start])
        visited[start] = True

        while queue:
            u = queue.popleft()
            order.append(u)
            for v, _ in self.adj_list[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return order

    def dfs(self, start):
        # пошук у глибину (DFS)
        visited = [False] * self.V
        order = []

        def _dfs(u):
            visited[u] = True
            order.append(u)
            for v, _ in self.adj_list[u]:
                if not visited[v]:
                    _dfs(v)

        _dfs(start)
        return order

    def has_cycle(self):
        # перевірка наявності циклів для неорієнтованого графа
        visited = [False] * self.V

        def _has_cycle_util(v, parent):
            visited[v] = True
            for neighbor, _ in self.adj_list[v]:
                if not visited[neighbor]:
                    if _has_cycle_util(neighbor, v):
                        return True
                elif parent != neighbor:
                    return True
            return False

        for i in range(self.V):
            if not visited[i]:
                if _has_cycle_util(i, -1):
                    return True
        return False

# -------------------------------------------------------------
# Завдання 1: Комп'ютерна мережа з 6 комп'ютерів
# -------------------------------------------------------------
def task1_network():
    print("=== Завдання 1: Граф комп'ютерної мережі ===")
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    n_map = {name: i for i, name in enumerate(nodes)}
    
    g = Graph(6)
    edges = [
        ('A', 'B', 10), ('A', 'C', 5),
        ('B', 'D', 15), ('B', 'E', 8),
        ('C', 'D', 12), ('C', 'F', 7),
        ('D', 'E', 20), ('D', 'F', 10),
        ('E', 'F', 18)
    ]
    for u, v, w in edges:
        g.add_edge(n_map[u], n_map[v], w)

    print("\nМатриця суміжності:")
    g.print_adj_matrix()

    print("\nСтепені вершин:")
    total_deg = 0
    for name in nodes:
        deg = g.degree(n_map[name])
        total_deg += deg
        print(f"deg({name}) = {deg}")
    print(f"Сума степенів: {total_deg}, 2*|E| = {2 * len(edges)} (теорема про рукостискання сходиться)")

    print(f"\nГраф зв'язний: {g.is_connected()}")
    
    # Дейкстра від вершини A
    dist = g.dijkstra(n_map['A'])
    print("\nНайкоротші шляхи від A (алгоритм Дейкстри):")
    for name in nodes:
        print(f"До {name}: відстань = {dist[n_map[name]]}")

    # візуалізація графа за допомогою networkx
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 5))
    nx.draw_networkx_nodes(G, pos, node_color='#4682b4', node_size=800)
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=2, edge_color='#666666')
    
    # підсвічуємо найкоротший шлях A -> C -> F
    path_edges = [('A', 'C'), ('C', 'F')]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3.5, edge_color='#d9534f')
    
    edge_labels = {(u, v): f"{w} Mbps" for u, v, w in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    
    plt.title("Граф мережі (найкоротший шлях A-C-F виділено червоним)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'task1_network_graph.png'), dpi=300)
    plt.close()
    print("Графік мережі збережено в plots/task1_network_graph.png")

# -------------------------------------------------------------
# Завдання 2: Графова модель соціальної мережі
# -------------------------------------------------------------
def task2_social():
    print("\n=== Завдання 2: Соціальна мережа ===")
    users = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
    idx = {name: i for i, name in enumerate(users)}
    
    # список зв'язків (хто кого додав у друзі)
    relations = [
        ('Alice', 'Bob', 2020),
        ('Alice', 'Charlie', 2019),
        ('Bob', 'Eve', 2021),
        ('Charlie', 'Diana', 2022),
        ('Diana', 'Eve', 2020),
        ('Eve', 'Alice', 2021)
    ]
    
    G = nx.DiGraph()
    for u in users:
        G.add_node(u)
    for u, v, year in relations:
        G.add_edge(u, v, year=year)

    print("Вхідні та вихідні степені користувачів:")
    for u in users:
        print(f"{u}: вихідний = {G.out_degree(u)}, вхідний (популярність) = {G.in_degree(u)}")

    print(f"\nЧи граф сильно зв'язний: {nx.is_strongly_connected(G)}")
    
    # друзі друзів для Alice (довжина шляху = 2)
    fof = set()
    for friend in G.successors('Alice'):
        for candidate in G.successors(friend):
            if candidate != 'Alice' and candidate not in G.successors('Alice'):
                fof.add(candidate)
    print(f"Друзі друзів для Alice (відстань 2): {fof}")

    # цикли
    cycles = list(nx.simple_cycles(G))
    print(f"Знайдені цикли в графі: {cycles}")

    # малюємо граф соцмережі
    pos = nx.circular_layout(G)
    plt.figure(figsize=(7, 5))
    nx.draw_networkx_nodes(G, pos, node_color='#5cb85c', node_size=1000)
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=2, edge_color='#333333', arrowsize=20, arrowstyle='->')
    edge_labels = {(u, v): str(d['year']) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    plt.title("Орієнтований граф соціальної мережі")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'task2_social_graph.png'), dpi=300)
    plt.close()
    print("Графік соцмережі збережено в plots/task2_social_graph.png")

# -------------------------------------------------------------
# Завдання 3: Залежності таблиць у базі даних
# -------------------------------------------------------------
def task3_database():
    print("\n=== Завдання 3: Аналіз залежностей таблиць БД ===")
    tables = ['Users', 'Orders', 'Products', 'OrderItems', 'Categories', 'ProductCategories']
    
    # Ребра від дочірньої таблиці до батьківської через Foreign Key
    fk_edges = [
        ('Orders', 'Users'),
        ('OrderItems', 'Orders'),
        ('OrderItems', 'Products'),
        ('ProductCategories', 'Products'),
        ('ProductCategories', 'Categories')
    ]
    
    G = nx.DiGraph()
    G.add_nodes_from(tables)
    G.add_edges_from(fk_edges)
    
    # Кореневі таблиці
    roots = [n for n in G.nodes if G.out_degree(n) == 0]
    leaves = [n for n in G.nodes if G.in_degree(n) == 0]
    
    print(f"Кореневі таблиці (не залежать від інших): {roots}")
    print(f"Листкові таблиці (на них ніхто не посилається): {leaves}")
    print(f"Чи є цикли в залежностях: {not nx.is_directed_acyclic_graph(G)}")
    
    # Порядок завантаження даних
    load_graph = G.reverse()
    load_order = list(nx.topological_sort(load_graph))
    print(f"Порядок завантаження даних (INSERT): {' -> '.join(load_order)}")
    print(f"Порядок безпечного видалення (DELETE): {' -> '.join(reversed(load_order))}")

    # візуалізація
    pos = nx.spring_layout(G, seed=12)
    plt.figure(figsize=(8, 5))
    nx.draw_networkx_nodes(G, pos, node_color='#f0ad4e', node_size=1200)
    nx.draw_networkx_labels(G, pos, font_color='black', font_weight='bold', font_size=9)
    nx.draw_networkx_edges(G, pos, width=2, edge_color='#555555', arrowsize=18, arrowstyle='->')
    plt.title("Граф залежностей таблиць БД (стрілка показує зовнішній ключ FK)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'task3_db_dependencies.png'), dpi=300)
    plt.close()
    print("Графік залежностей БД збережено в plots/task3_db_dependencies.png")

# -------------------------------------------------------------
# Завдання 4: Тестування класу Graph (BFS, DFS, цикли)
# -------------------------------------------------------------
def task4_testing():
    print("\n=== Завдання 4: Тестування класу Graph ===")
    g = Graph(6)
    edges = [
        (0, 1, 10), (0, 2, 5),
        (1, 3, 15), (1, 4, 8),
        (2, 3, 12), (2, 5, 7),
        (3, 4, 20), (3, 5, 10),
        (4, 5, 18)
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)

    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    print(f"Обхід графа в ширину (BFS) від A: {[nodes[i] for i in g.bfs(0)]}")
    print(f"Обхід графа в глибину (DFS) від A: {[nodes[i] for i in g.dfs(0)]}")
    print(f"Чи містить граф цикли: {g.has_cycle()}")

if __name__ == '__main__':
    task1_network()
    task2_social()
    task3_database()
    task4_testing()
