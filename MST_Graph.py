import heapq

class Graph:
    # 初始化
    def __init__(self):
        self.graph = {}
        self.keys = []

    # 添加边
    # 分为初始化和扩展2个环节
    def add_edge(self, u, v, cost):
        if u in self.graph:
            self.graph[u].append((v,cost))
        else:
            self.graph[u] = [(v,cost)]
            self.keys.append(u)
        if v in self.graph:
            self.graph[v].append((u,cost))
        else:
            self.graph[v] = [(u,cost)]
            self.keys.append(v)

    def __str__(self):
        result = ""
        for node, (neighbors, cost) in self.graph.items():
            result += f"{node}: {neighbors}{cost}\n"
        return result
       

    def prim(self):
        min_spanning_tree = []
        start_node = self.keys[0] #起始节点
        visited = set([start_node])
        edges = [
            (cost, start_node, next_node)
            for next_node, cost in self.graph[start_node]
        ] #对于start_node所连接的结点next_node及其距离cost
        # 按照前后顺序建立堆heap
        heapq.heapify(edges)

        while edges:
            # heapq.heappop()是从堆中弹出并返回最小的值
            cost, start, next_node = heapq.heappop(edges)
            # 如果该结点没有被访问过
            if next_node not in visited:
                visited.add(next_node)
                min_spanning_tree.append((start, next_node, cost))

                #对于next_node所连接的结点neighbor及其距离cost
                for neighbor, cost in self.graph[next_node]:
                    if neighbor not in visited:
                        # heapq.heappush()是往堆中添加新值，此时自动建立了小根堆
                        # 小根堆为堆顶元素最小的堆
                        heapq.heappush(edges, (cost, next_node, neighbor))

        return min_spanning_tree