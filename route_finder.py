# -*- coding: utf-8 -*-

import heapq
from collections import deque

graph = {
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Zerind': {'Oradea': 71, 'Arad': 75},
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Sibiu': {'Oradea': 151, 'Arad': 140, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Pitesti': 97, 'Craiova': 146},
    'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

def breadth_first_search(graph, start, goal):
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        (vertex, path) = queue.popleft()
        
        for neighbor in sorted(graph[vertex].keys()):
            if neighbor == goal:
                final_path = path + [neighbor]
               
                cost = sum(graph[final_path[i]][final_path[i+1]] for i in range(len(final_path)-1))
                return final_path, cost
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
                
    return None, float('inf')

def uniform_cost_search(graph, start, goal):
    """Realiza a busca de custo uniforme."""
    visited = set()
    priority_queue = [(0, start, [start])]
    
    while priority_queue:
        (cost, current_node, path) = heapq.heappop(priority_queue)
        
        if current_node == goal:
            return path, cost
            
        if current_node not in visited:
            visited.add(current_node)
            
            for neighbor, weight in graph[current_node].items():
                if neighbor not in visited:
                    new_cost = cost + weight
                    heapq.heappush(priority_queue, (new_cost, neighbor, path + [neighbor]))
                    
    return None, float('inf')

def depth_first_search(graph, start, goal):
    """Realiza a busca em profundidade (iterativa)."""
    visited = {start}
    stack = [(start, [start])]
    
    while stack:
        (vertex, path) = stack.pop()
        
        if vertex == goal:
             return path, sum(graph[path[i]][path[i+1]] for i in range(len(path)-1))

        for neighbor in sorted(graph[vertex].keys(), reverse=True):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
                
    return None, float('inf')

def depth_limited_search(graph, start, goal, limit):
    """Realiza a busca em profundidade limitada."""
    def recursive_dls(node, path, current_limit):
        if node == goal:
            return path, sum(graph[path[i]][path[i+1]] for i in range(len(path)-1))
        elif current_limit == 0:
            return None, float('inf')
            
        for neighbor in graph[node]:
            if neighbor not in path:
                result_path, result_cost = recursive_dls(neighbor, path + [neighbor], current_limit - 1)
                if result_path is not None:
                    return result_path, result_cost
                
        return None, float('inf')
        
    return recursive_dls(start, [start], limit)

def iterative_deepening_search(graph, start, goal):
    """Realiza a busca de aprofundamento iterativo."""
    for depth in range(len(graph)):
        path, cost = depth_limited_search(graph, start, goal, depth)
        if path is not None:
            return path, cost
    return None, float('inf')
    
def bidirectional_search(graph, start, goal):
    """Realiza busca bidirecional (implementação básica)."""
    if start == goal:
        return [start], 0

    queue_start = deque([(start, [start])])
    visited_start = {start: [start]}
    cost_start = {start: 0}

    queue_goal = deque([(goal, [goal])])
    visited_goal = {goal: [goal]}
    cost_goal = {goal: 0}

    meeting_node = None
    min_total_cost = float('inf')

    while queue_start and queue_goal:
        if queue_start:
            current_start, path_start = queue_start.popleft()
            current_cost_start = cost_start[current_start]

            if current_start in visited_goal:
                path_goal_rev = visited_goal[current_start]
                combined_path = path_start + path_goal_rev[::-1][1:]
                total_cost = current_cost_start + cost_goal[current_start]
                if total_cost < min_total_cost:
                     min_total_cost = total_cost
                     meeting_node = current_start

            if current_cost_start >= min_total_cost:
                 continue

            for neighbor, weight in graph[current_start].items():
                if neighbor not in visited_start:
                    new_path = path_start + [neighbor]
                    visited_start[neighbor] = new_path
                    cost_start[neighbor] = current_cost_start + weight
                    queue_start.append((neighbor, new_path))
                elif current_cost_start + weight < cost_start[neighbor]:
                     cost_start[neighbor] = current_cost_start + weight
                     visited_start[neighbor] = path_start + [neighbor]

        if queue_goal:
            current_goal, path_goal_rev = queue_goal.popleft()
            current_cost_goal = cost_goal[current_goal]

            if current_goal in visited_start:
                path_start_to_meet = visited_start[current_goal]
                combined_path = path_start_to_meet + path_goal_rev[::-1][1:]
                total_cost = cost_start[current_goal] + current_cost_goal
                if total_cost < min_total_cost:
                    min_total_cost = total_cost
                    meeting_node = current_goal

            if current_cost_goal >= min_total_cost:
                 continue

            for neighbor, weight in graph[current_goal].items():
                if neighbor not in visited_goal:
                    new_path_rev = path_goal_rev + [neighbor]
                    visited_goal[neighbor] = new_path_rev
                    cost_goal[neighbor] = current_cost_goal + weight
                    queue_goal.append((neighbor, new_path_rev))
                elif current_cost_goal + weight < cost_goal[neighbor]:
                    cost_goal[neighbor] = current_cost_goal + weight
                    visited_goal[neighbor] = path_goal_rev + [neighbor]

    if meeting_node:
        final_path = visited_start[meeting_node] + visited_goal[meeting_node][::-1][1:]
        return final_path, min_total_cost
    else:
        return None, float('inf')

heuristic = {
    'Oradea': 0, 'Zerind': 0, 'Arad': 0, 'Timisoara': 0, 'Lugoj': 0,
    'Mehadia': 0, 'Drobeta': 0, 'Sibiu': 0, 'Rimnicu Vilcea': 0,
    'Craiova': 0, 'Fagaras': 0, 'Pitesti': 0, 'Bucharest': 0,
    'Giurgiu': 0, 'Urziceni': 0, 'Hirsova': 0, 'Eforie': 0,
    'Vaslui': 0, 'Iasi': 0, 'Neamt': 0
}

def greedy_search(graph, start, goal, heuristic):
    """Realiza a busca gulosa."""
    visited = set()
    priority_queue = [(heuristic[start], start, [start])]
    cost_so_far = {start: 0}

    while priority_queue:
        (_, current_node, path) = heapq.heappop(priority_queue)

        if current_node == goal:
            actual_cost = sum(graph[path[i]][path[i+1]] for i in range(len(path)-1))
            return path, actual_cost

        if current_node not in visited:
            visited.add(current_node)

            for neighbor, weight in graph[current_node].items():
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    heapq.heappush(priority_queue, (heuristic[neighbor], neighbor, new_path))
                    cost_so_far[neighbor] = cost_so_far[current_node] + weight

    return None, float('inf')

def a_star_search(graph, start, goal, heuristic):
    """Realiza a busca A*."""
    visited = set()
    priority_queue = [(heuristic[start], 0, start, [start])]
    cost_so_far = {start: 0}

    while priority_queue:
        (estimated_cost, cost, current_node, path) = heapq.heappop(priority_queue)
        
        if current_node == goal:
            return path, cost
            
        if current_node in visited:
             continue
        visited.add(current_node)

        for neighbor, weight in graph[current_node].items():
            new_cost = cost + weight
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic[neighbor]
                heapq.heappush(priority_queue, (priority, new_cost, neighbor, path + [neighbor]))
                
    return None, float('inf')

def get_city_by_input(cities, prompt):
    cities_lower = {city.lower(): city for city in cities}
    cities_list = sorted(cities)
    
    while True:
        print("\nCidades disponíveis:")
        for i, city in enumerate(cities_list, 1):
            print(f"{i} => {city}")
            
        user_input = input(f"\n{prompt}").strip()
        
        try:
            idx = int(user_input)
            if 1 <= idx <= len(cities_list):
                return cities_list[idx-1]
            print("Número inválido. Por favor, escolha um número da lista.")
        except ValueError:
            city_input = user_input.lower()
            if city_input in cities_lower:
                return cities_lower[city_input]
            print("Entrada inválida. Digite o número ou nome da cidade.")

def main():
    print("Algoritmo de Busca de Rotas")
    
    start_node = get_city_by_input(graph.keys(), "Digite o número ou nome da cidade de origem: ")
    goal_node = get_city_by_input(graph.keys(), "Digite o número ou nome da cidade de destino: ")
    
    if start_node.lower() == goal_node.lower():
        print("Origem e destino são iguais!")
        return

    print("\n--- Resultados das Buscas ---")

    path_bfs, cost_bfs = breadth_first_search(graph, start_node, goal_node)
    print("\n1.1 Busca em Largura (BFS):")
    if path_bfs:
        print(f"  Caminho: {' -> '.join(path_bfs)}")
        print(f"  Custo: {cost_bfs}")
    else:
        print("  Caminho não encontrado.")

    path_ucs, cost_ucs = uniform_cost_search(graph, start_node, goal_node)
    print("\n1.2 Busca de Custo Uniforme (UCS):")
    if path_ucs:
        print(f"  Caminho: {' -> '.join(path_ucs)}")
        print(f"  Custo: {cost_ucs}")
    else:
        print("  Caminho não encontrado.")

    path_dfs, cost_dfs = depth_first_search(graph, start_node, goal_node)
    print("\n1.3 Busca em Profundidade (DFS):")
    if path_dfs:
        print(f"  Caminho: {' -> '.join(path_dfs)}")
        print(f"  Custo: {cost_dfs}")
    else:
        print("  Caminho não encontrado.")

    limit = 3
    print(f"\n1.4 Busca em Profundidade Limitada (DLS) (Limite={limit}):")
    path_dls, cost_dls = depth_limited_search(graph, start_node, goal_node, limit)
    if path_dls:
        print(f"  Caminho: {' -> '.join(path_dls)}")
        print(f"  Custo: {cost_dls}")
    else:
        print(f"  Caminho não encontrado dentro do limite {limit}.")
        
    path_ids, cost_ids = iterative_deepening_search(graph, start_node, goal_node)
    print("\n1.5 Busca de Aprofundamento Iterativo (IDS):")
    if path_ids:
        print(f"  Caminho: {' -> '.join(path_ids)}")
        print(f"  Custo: {cost_ids}")
    else:
        print("  Caminho não encontrado.")
        
    path_bi, cost_bi = bidirectional_search(graph, start_node, goal_node)
    print("\n1.6 Busca Bidirecional:")
    if path_bi:
        print(f"  Caminho: {' -> '.join(path_bi)}")
        print(f"  Custo: {cost_bi}")
    else:
        print("  Caminho não encontrado.")

    print("\n--- Buscas Heurísticas --- (Usando heurística de exemplo - h(n)=0)")

    path_greedy, cost_greedy = greedy_search(graph, start_node, goal_node, heuristic)
    print("\n2.1 Busca Gulosa:")
    if path_greedy:
        print(f"  Caminho: {' -> '.join(path_greedy)}")
        print(f"  Custo: {cost_greedy}")
    else:
        print("  Caminho não encontrado.")

    path_astar, cost_astar = a_star_search(graph, start_node, goal_node, heuristic)
    print("\n2.2 Busca A*:")
    if path_astar:
        print(f"  Caminho: {' -> '.join(path_astar)}")
        print(f"  Custo: {cost_astar}")
    else:
        print("  Caminho não encontrado.")

if __name__ == "__main__":
    main()