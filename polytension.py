import matplotlib.pyplot as plt
import numpy as np

#  Polynomial time complexity
#  Worse case: O(n^4)
#  Best case: O(n^3)

NODES = 100
SQRT_2 = np.sqrt(2)


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.adj_list = {}


class Util:
    @staticmethod
    def dist(a, b):
        return np.sqrt(((a.x - b.x) ** 2) + ((a.y - b.y) ** 2))

    @staticmethod
    def get_line(a, b):
        x = np.linspace(a.x, b.x, 2)
        return x, (((b.y - a.y) / (b.x - a.x)) * (x - a.x)) + a.y

    @staticmethod
    def on_segment(p, q, r):
        return min(p.x, r.x) <= q.x <= max(p.x, r.x) and min(p.y, r.y) <= q.y <= max(p.y, r.y)

    @staticmethod
    def orientation(p, q, r):
        val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y))

        if val == 0:
            return 0

        return 1 if val > 0 else 2

    @staticmethod
    def do_intersect(p1, q1, p2, q2):
        o1 = Util.orientation(p1, q1, p2)
        o2 = Util.orientation(p1, q1, q2)
        o3 = Util.orientation(p2, q2, p1)
        o4 = Util.orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and Util.on_segment(p1, p2, q1):
            return True
        if o2 == 0 and Util.on_segment(p1, q2, q1):
            return True
        if o3 == 0 and Util.on_segment(p2, p1, q2):
            return True
        if o4 == 0 and Util.on_segment(p2, q1, q2):
            return True

        return False


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def insert_node(self, new_node):
        for node in self.nodes:
            dist = Util.dist(node, new_node)

            node.adj_list[new_node] = dist
            new_node.adj_list[node] = dist

        self.nodes.append(new_node)

    def dist(self, a, b):
        if a == b:
            return 0
        return a.adj_list[b]

    def edge_dist(self, edge):
        return self.dist(edge.a, edge.b)

    def __len__(self):
        return len(self.nodes)


class TSP:
    def __init__(self, graph):
        self.graph = graph
        self.source = graph.nodes[0]
        self.path = [self.source]
        self.best_path = []
        self.best_dist = -1

    def draw(self):
        cm = plt.get_cmap('rainbow')

        for i in range(len(self.best_path) - 1):
            a = self.best_path[i]
            b = self.best_path[i + 1]
            x, y = Util.get_line(a, b)
            dist = self.graph.dist(a, b)
            plt.plot(x, y, color="g")

    def solve_recursive(self, dist):
        if len(self.path) == len(self.graph):
            self.path.append(self.source)
            dist += self.graph.dist(self.path[-1], self.path[-2])

            if dist < self.best_dist or self.best_dist == -1:
                self.best_path = self.path.copy()
                self.best_dist = dist

            self.path.pop()
        else:
            for adj in self.path[-1].adj_list:
                if adj in self.path:
                    continue

                self.path.append(adj)
                self.solve_recursive(dist + self.graph.dist(self.path[-1], self.path[-2]))
                self.path.pop()

            if self.path[-1] == self.source:
                return

    def solve(self):
        self.__init__(self.graph)
        self.solve_recursive(0)
        return self.best_path


class Edge:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.m = (self.b.y - self.a.y) / (self.b.x - self.a.x)

    def y(self, x):
        return (self.m * (x - self.a.x)) + self.a.y

    def compare(self, node):
        return self.y(node.x) > node.y


class PolyTension:
    def __init__(self, graph):
        self.graph = graph
        self.com = Node(np.average([n.x for n in graph.nodes]), np.average([n.y for n in graph.nodes]))
        self.com_dist = {node: Util.dist(node, self.com) for node in graph.nodes}
        self.outer_poly = []
        self.inner_poly = []
        self.inner_nodes = []
    
    def draw(self):
        for edge in self.inner_poly:
            x, y = Util.get_line(edge.a, edge.b)
            plt.plot(x, y, color="r")

    def find_outer_edge(self, curr_node, visited_nodes):
        for node in self.graph.nodes:
            if node == curr_node or node in visited_nodes:
                continue

            edge = Edge(curr_node, node)
            target_cmp = None
            is_outer = True
            for n in self.graph.nodes:
                if n == curr_node or n == node:
                    continue

                cmp = edge.compare(n)
                if target_cmp is None:
                    target_cmp = cmp
                else:
                    if cmp != target_cmp:
                        is_outer = False
                        break

            if is_outer:
                return node, edge
        return None, None

    def get_far_node(self, nodes):
        far_node = None
        max_dist = -1.0
        for node in nodes:
            if self.com_dist[node] > max_dist:
                far_node = node
                max_dist = self.com_dist[node]
        return far_node

    def build_outer_poly(self):
        far_node = self.get_far_node(self.graph.nodes)

        visited_nodes = [far_node]

        curr_node, edge = self.find_outer_edge(far_node, visited_nodes)
        self.outer_poly.append(edge)
        while curr_node is not None:
            next_node, edge = self.find_outer_edge(curr_node, visited_nodes)
            if next_node is None:
                break

            self.outer_poly.append(edge)
            visited_nodes.append(curr_node)
            curr_node = next_node

        self.outer_poly.append(Edge(curr_node, far_node))
        visited_nodes.append(curr_node)

        for node in self.graph.nodes:
            if node not in visited_nodes:
                self.inner_nodes.append(node)

        return self.outer_poly

    def get_best_step(self):
        best_edge = None
        best_node = None
        best_score = -1

        for edge in self.inner_poly:
            for node in self.inner_nodes:
                score = self.graph.edge_dist(edge) / ((self.graph.dist(node, edge.a) + self.graph.dist(node, edge.b)) ** 1.3)
                # score = np.power(0.6 + score, SQRT_2 - self.graph.edge_dist(edge))

                if score > best_score:
                    skip = False

                    for e in self.inner_poly:
                        connected_a = False
                        connected_b = False

                        if e == edge:
                            continue
                        elif e.a == edge.a or e.b == edge.a:
                            connected_a = True
                        elif e.a == edge.b or e.b == edge.b:
                            connected_b = True

                        if not connected_a and Util.do_intersect(e.a, e.b, edge.a, node) or not connected_b and Util.do_intersect(e.a, e.b, edge.b, node):
                            skip = True
                            break

                    if skip:
                        break

                    best_edge = edge
                    best_node = node
                    best_score = score

        return best_edge, best_node

    def collapse(self, tsp):
        self.inner_poly = self.outer_poly

        while self.inner_nodes:
            # Uncomment lines below to show step-by-step run through

            # tsp.draw()
            # self.draw()
            # plt.plot([n.x for n in self.graph.nodes], [n.y for n in self.graph.nodes], 'ko')
            # plt.show()
            # plt.clf()

            best_edge, best_node = self.get_best_step()

            self.inner_poly.remove(best_edge)
            self.inner_poly.append(Edge(best_edge.a, best_node))
            self.inner_poly.append(Edge(best_edge.b, best_node))

            self.inner_nodes.remove(best_node)

        return self.inner_poly


def main():
    graph = Graph()

    for i in range(NODES):
        graph.insert_node(Node(np.random.random(), np.random.random()))

    tsp = TSP(graph)
    # Uncomment to run recursive tsp algo
    # tsp.solve()

    pt = PolyTension(graph)
    pt.build_outer_poly()
    pt.collapse(tsp)

    # Uncomment to show recursive tsp algo
    # tsp.draw()
    pt.draw()

    plt.plot([n.x for n in graph.nodes], [n.y for n in graph.nodes], 'ko')

    plt.show()


while True:
    main()
