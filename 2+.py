import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from io import BytesIO
from PIL import Image
import re

def bfs(graph, start, goal):
    queue = deque([(start, [start])])
    visited = set([start])
    count = 1
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path, count
        for neighbor, _ in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                count += 1
                queue.append((neighbor, path + [neighbor]))
    return None, count

def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    count = 0
    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        count += 1
        if node == goal:
            return path, count
        for neighbor, _ in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None, count

def draw_graph(graph, path):
    G = nx.DiGraph()
    for u in graph:
        for v, w in graph[u]:
            G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, edge_color='gray', arrows=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    if path:
        edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange')

    buf = BytesIO()
    plt.savefig(buf, format='PNG')
    plt.close()
    buf.seek(0)
    image = Image.open(buf)
    return image

def find_path(edge_text, start, end, algorithm):
    graph = {}
    start = start.strip().upper()
    end = end.strip().upper()
    has_weight = False

    for line in edge_text.strip().split("\n"):
        line = line.strip().upper()
        u = v = None
        w = 1.0  # mặc định trọng số là 1

        match = re.match(r'(\w+)\s*-->\s*\|(\d+(\.\d+)?)\|\s*(\w+)', line)
        if match:
            u, w, v = match.group(1), float(match.group(2)), match.group(4)
            has_weight = True
        elif '-->' in line:
            parts = [x.strip() for x in line.split('-->')]
            if len(parts) == 2:
                u, v = parts
                w = 1.0
        else:
            parts = line.split()
            if len(parts) == 3:
                u, v, w = parts[0], parts[1], float(parts[2])
                has_weight = True
            elif len(parts) == 2:
                u, v = parts
                w = 1.0

        if u and v:
            graph.setdefault(u, []).append((v, w))
            graph.setdefault(v, [])

    if algorithm == "BFS":
        path, count = bfs(graph, start, end)
    else:
        path, count = dfs(graph, start, end)

    total_weight = 0
    if path and has_weight:
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            for neighbor, w in graph[u]:
                if neighbor == v:
                    total_weight += w
                    break

    img = draw_graph(graph, path)

    if path:
        result = f"Đường đi: {' -> '.join(path)}\nSố nút đã thăm: {count}"
        if has_weight:
            result += f"\nTổng trọng số: {total_weight}"
    else:
        result = "Không tìm thấy đường đi!"

    return result, img

iface = gr.Interface(
    fn=find_path,
    inputs=[
        gr.Textbox(label="Danh sách cạnh", lines=15, placeholder="S -->|2| A\nA --> B\nB C 3\n..."),
        gr.Textbox(label="Đỉnh bắt đầu"),
        gr.Textbox(label="Đỉnh kết thúc"),
        gr.Radio(["BFS", "DFS"], label="Thuật toán", value="BFS")
    ],
    outputs=[
        gr.Textbox(label="Kết quả"),
        gr.Image(label="Hình đồ thị")
    ],
    title="Tìm đường đi trên đồ thị bằng BFS/DFS"
)

iface.launch(share=False)
