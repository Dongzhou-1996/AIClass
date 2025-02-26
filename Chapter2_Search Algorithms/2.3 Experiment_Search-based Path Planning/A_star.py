import time
import numpy as np
import math

HEURISTIC_FUNC_EUCLIDEAN = 1
HEURISTIC_FUNC_MANHATTAN = 2
heuristic_func = 1


# A*规划类
class AstarPlanner:
    # A*节点
    class AstarNode:
        def __init__(self):
            self.x = 0
            self.y = 0
            self.value_f = 0
            self.value_g = 0
            self.value_h = 0
            self.father = None

        # 节点启发函数
        def h_calculate(self, goal):
            dx = goal.x - self.x
            dy = goal.y - self.y
            if heuristic_func == HEURISTIC_FUNC_EUCLIDEAN:
                self.value_h = math.sqrt(dx * dx + dy * dy)
            else:
                self.value_h = math.fabs(dx) + math.fabs(dy)
            self.value_f = self.value_g + self.value_h

    def __init__(self):
        self.close_list = []
        self.open_list = []
        self.map = None
        self.map_x_size = 0
        self.map_y_size = 0

    # 地图导入函数
    def import_map(self, map_mat):
        self.map = map_mat
        self.map_x_size = map_mat.shape[0]
        self.map_y_size = map_mat.shape[1]

    # 规划函数
    def plan(self, start, goal):
        # 开放列表与关闭列表清空
        self.open_list = []
        self.close_list = []
        # 起点节点与目标节点
        start_node = self.AstarNode()
        start_node.x = int(start[0])
        start_node.y = int(start[1])
        goal_node = self.AstarNode()
        goal_node.x = int(goal[0])
        goal_node.y = int(goal[1])
        start_node.h_calculate(goal=goal_node)
        # 起点节点存入开放列表
        self.open_list.append(start_node)
        while True:
            # 选择当前最优节点
            current_node_index = self.choose_optimal_node()
            current_node = self.open_list[current_node_index]
            # 如果以到达终点 则生成路径
            if (current_node.x == goal_node.x) and \
                    (current_node.y == goal_node.y):
                path = []
                path.append([current_node.x, current_node.y])
                while True:
                    current_node = current_node.father
                    if current_node == None:
                        break
                    path.append([current_node.x, current_node.y])
                path.reverse()
                print("Path: ", path)
                print("Open list size: ", len(self.open_list))
                print("Close list size: ", len(self.close_list))
                return path
            # 节点扩展
            DXs = [-1, 0, 1]
            DYs = [-1, 0, 1]
            for dx in DXs:
                for dy in DYs:
                    if (dx == 0) and (dy == 0):
                        continue
                    if (current_node.x + dx >= self.map_x_size) \
                            or (current_node.x + dx < 0):
                        continue
                    if (current_node.y + dy >= self.map_y_size) \
                            or (current_node.y + dy < 0):
                        continue
                    if self.map[current_node.x + dx, \
                            current_node.y + dy] == 1:
                        continue
                    temp_node = self.AstarNode()
                    temp_node.father = current_node
                    temp_node.x = current_node.x + dx
                    temp_node.y = current_node.y + dy
                    temp_node.value_g = current_node.value_g \
                                        + math.sqrt(dx * dx + dy * dy)
                    temp_node.h_calculate(goal_node)
                    open_index = self.find_node(temp_node, self.open_list)
                    close_index = self.find_node(temp_node, self.close_list)
                    if open_index != -1:
                        if temp_node.value_f < self.open_list \
                                [open_index].value_f:
                            self.open_list[open_index] = temp_node
                    elif close_index != -1:
                        if temp_node.value_f < self.close_list \
                                [close_index].value_f:
                            self.close_list[close_index] = temp_node
                    else:
                        self.open_list.append(temp_node)
            self.open_list.pop(current_node_index)
            self.close_list.append(current_node)

    # 最优节点选取函数
    def choose_optimal_node(self):
        optimal_node_index = -1
        optimal_node_f = 1e6
        for index, node in enumerate(self.open_list):
            if node.value_f < optimal_node_f:
                optimal_node_index = index
                optimal_node_f = node.value_f
        return optimal_node_index

    # 节点搜索函数
    def find_node(self, node, node_list):
        node_index = -1
        for index, list_node in enumerate(node_list):
            if (list_node.x == node.x) and \
                    (list_node.y == node.y):
                node_index = index
                break
        return node_index


# 主函数
if __name__ == "__main__":
    planner = AstarPlanner()
    map_mat = np.zeros(shape=(20, 20), dtype=bool)
    map_mat[5:9, 5:11] = 1
    map_mat[12:17, 9:13] = 1
    planner.import_map(map_mat)
    start_time = time.perf_counter()
    path = planner.plan([3, 3], [15, 15])
    finish_time = time.perf_counter()
    planning_time = finish_time - start_time
print("Planning time: %fs" % planning_time)
