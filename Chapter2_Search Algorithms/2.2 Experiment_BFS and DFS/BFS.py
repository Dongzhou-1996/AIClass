from collections import deque

# 定义图，使用字典表示邻接表和边的代价（权值）
graph = {
    1: {2: 4, 3: 3},  # 城市1到城市2的权值是4，城市1到城市3的权值是3
    2: {1: 4, 4: 4, 5: 5},  # 城市2到城市1的权值是4，城市2到城市4的权值是4,城市2到城市5的权值是5
    3: {1: 3, 4: 2},  # 城市3到城市1的权值是3，城市3到城市4的权值是2
    4: {2: 4, 3: 2, 5: 3},  # 城市4到城市2的权值是4, 城市4到城市3的权值是2,城市4到城市5的权值是3
    5: {2: 5, 4: 3}  # 城市5到城市2的权值是5, 城市5到城市4的权值是3
}


def BFS_BEST_Paths(graph, start, goal):
    # 用一个队列来存储当前的路径及其代价值，队列中每个元素是一个元组：(当前城市, 当前路径, 当前路径代价)
    queue = deque([(start, [start], 0)])
    all_paths = []  # 存储所有路径及其代价

    while queue:
        current, path, cost = queue.popleft()
        # 如果到达目标城市，则记录该路径及其代价
        if current == goal:
            all_paths.append((path, cost))
        # 遍历当前城市的所有邻接城市
        for neighbor, weight in graph.get(current, {}).items():
            if neighbor not in path:  # 防止重复走同一条城市
                queue.append((neighbor, path + [neighbor], cost + weight))

    return all_paths


# 获取从城市1到城市5的所有路径及其代价值
start_city = 1
end_city = 5
all_paths = BFS_BEST_Paths(graph, start_city, end_city)

# 打印所有路径及其代价
print("所有可能路径及其代价值：")
for path, cost in all_paths:
    print(f"路径: {path}, 代价: {cost}")

# 找到代价最小的路径
if all_paths:
    min_path, min_cost = min(all_paths, key=lambda x: x[1])
    print(f"\n最优路径: {min_path}, 最优代价: {min_cost}")
else:
    print("没有从城市1到城市5的路径")
