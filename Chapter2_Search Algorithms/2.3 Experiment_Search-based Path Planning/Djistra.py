import time
import numpy as np
import math

class DijkstraPlanner:
    class DijkstraNode:
        def __init__(self):
            self.x = 0
            self.y = 0
            self.value_g = 0
            self.father = None

    def __init__(self):
        self.close_list = []  # 关闭列表
        self.open_list = []  # 开放列表
        self.map = None  # 地图矩阵
        self.map_x_size = 0  # 地图x轴大小
        self.map_y_size = 0  # 地图y轴大小

    def import_map(self, map_mat):
        self.map = map_mat
        self.map_x_size = map_mat.shape[0]
        self.map_y_size = map_mat.shape[1]

    def plan(self, start, goal):
        self.open_list = []  # 清空开放列表
        self.close_list = []  # 清空关闭列表

        start_node = self.DijkstraNode()
        start_node.x = int(start[0])
        start_node.y = int(start[1])
        goal_node = self.DijkstraNode()
        goal_node.x = int(goal[0])
        goal_node.y = int(goal[1])

        self.open_list.append(start_node)  # 将起点加入开放列表

        while True:
            current_node_index = self.choose_optimal_node()  # 选择当前最优节点
            current_node = self.open_list[current_node_index]

            if (current_node.x == goal_node.x) and (current_node.y == goal_node.y):
                path = []
                path.append([current_node.x, current_node.y])
                while True:
                    current_node = current_node.father
                    if current_node is None:
                        break
                    path.append([current_node.x, current_node.y])
                path.reverse()
                print("Path: ", path)
                print("Open list size: ", len(self.open_list))
                print("Close list size: ", len(self.close_list))
                return path

            DXs = [-1, 0, 1]
            DYs = [-1, 0, 1]
            for dx in DXs:
                for dy in DYs:
                    if (dx == 0) and (dy == 0):
                        continue
                    if (current_node.x + dx >= self.map_x_size) or (current_node.x + dx < 0):
                        continue
                    if (current_node.y + dy >= self.map_y_size) or (current_node.y + dy < 0):
                        continue
                    if self.map[current_node.x + dx, current_node.y + dy] == 1:
                        continue
                    temp_node = self.DijkstraNode()
                    temp_node.father = current_node
                    temp_node.x = current_node.x + dx
                    temp_node.y = current_node.y + dy
                    temp_node.value_g = current_node.value_g + math.sqrt(dx * dx + dy * dy)
                    open_index = self.find_node(temp_node, self.open_list)
                    close_index = self.find_node(temp_node, self.close_list)
                    if open_index != -1:
                        if temp_node.value_g < self.open_list[open_index].value_g:
                            self.open_list[open_index] = temp_node
                    elif close_index != -1:
                        if temp_node.value_g < self.close_list[close_index].value_g:
                            self.close_list[close_index] = temp_node
                    else:
                        self.open_list.append(temp_node)
            self.open_list.pop(current_node_index)
            self.close_list.append(current_node)

    def choose_optimal_node(self):
        optimal_node_index = -1
        optimal_node_g = 1e6
        for index, node in enumerate(self.open_list):
            if node.value_g < optimal_node_g:
                optimal_node_index = index
                optimal_node_g = node.value_g
        return optimal_node_index

    def find_node(self, node, node_list):
        node_index = -1
        for index, list_node in enumerate(node_list):
            if (list_node.x == node.x) and (list_node.y == node.y):
                node_index = index
                break
        return node_index


if __name__ == "__main__":
    planner = DijkstraPlanner()
    map_mat = np.zeros(shape=(20, 20), dtype=bool)
    map_mat[5:9, 5:11] = 1
    map_mat[12:17, 9:13] = 1
    planner.import_map(map_mat)
    start_time = time.perf_counter()
    path = planner.plan([3, 3], [15, 15])
    finish_time = time.perf_counter()
    planning_time = finish_time - start_time
    print("Planning time: %fs" % planning_time)
