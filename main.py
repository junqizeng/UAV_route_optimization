import matplotlib.pyplot as plt
import random
import numpy as np
import copy
from config import *

"""
流程：
- 每个时间刻开始，更新未完成任务时间，重新分配任务集合
- 对每个配送中心随机生成若干任务，分配任务集合
- 根据任务集合进行路径规划
- 规划执行完毕，该时间刻结束
"""


class Route:
    def __init__(self, drop_off_num):
        self.route = []
        self.cargo_dist_list = [0 for _ in range(drop_off_num)]
        self.distance = 0
        self.unvisited_nodes = set([idx for idx in range(drop_off_num)])
        self.last_pos = -1


class DistCenterPoint:
    def __init__(self, pos):
        """
        unsolved_task_list：待完成任务集，其中每个任务项属性意义：[卸货点序号，剩余时间]
        :param pos:
        """
        self.pos = pos
        self.drop_off_point_list = []
        self.distance_matrix = np.array([[0]], dtype=float)
        self.unsolved_task_list = []
        self.essential_task_list = []

    def add_drop_off_point(self, drop_off_point):
        cur_drop_off_point_num = len(self.drop_off_point_list)
        # 更新距离矩阵
        self.distance_matrix = np.pad(self.distance_matrix, ((0, 1), (0, 1)), 'constant')
        self.distance_matrix[0][cur_drop_off_point_num + 1] = np.linalg.norm(
            np.array(drop_off_point) - np.array(self.pos))
        self.distance_matrix[cur_drop_off_point_num + 1][0] = np.linalg.norm(
            np.array(drop_off_point) - np.array(self.pos))

        for i in range(cur_drop_off_point_num):
            cur_point = self.drop_off_point_list[i]
            self.distance_matrix[i + 1][cur_drop_off_point_num + 1] = np.linalg.norm(
                np.array(drop_off_point) - np.array(cur_point))
            self.distance_matrix[cur_drop_off_point_num + 1][i + 1] = np.linalg.norm(
                np.array(drop_off_point) - np.array(cur_point))

        # 添加新的卸货点
        self.drop_off_point_list.append(drop_off_point)
        self.essential_task_list.append(0)

    def update_unsolved_task_set(self):
        """
        每个时间刻开始，更新未解决任务集合时间，并重新分配任务集合
        :return:
        """
        for task in self.unsolved_task_list:
            task[1] -= TIME_INTERVAL
            assert task[1] > 0
            if task[1] <= MIN_REMAIN_TIME:
                self.essential_task_list[task[0]] += 1
                self.unsolved_task_list.remove(task)

    def generate_new_task(self):
        """
        每个卸货点随机生成 0 - MAX_TASK_NUM 个订单
        任务等级分布为：50%紧急任务（30min），30%较紧急任务（90min），20%一般任务（180min）
        :return:
        """
        for idx in range(len(self.drop_off_point_list)):
            new_task_num = random.randint(0, MAX_TASK_NUM)
            for _ in range(new_task_num):
                task_level = random.randint(0, 10)
                if task_level < 5:
                    self.unsolved_task_list.append([idx, 30])
                elif task_level < 8:
                    self.unsolved_task_list.append([idx, 90])
                else:
                    self.unsolved_task_list.append([idx, 180])

    def route_optimize(self):
        """
        根据必须完成任务集规划任务路径
        使用限制的dfs搜索路径，尽可能最大卸货量，搜索出来的路径按路程排列，取最短路程安排后，更新任务集
        直到必须任务集为空
        :return:
        """
        print(f'总重要任务数： {sum(self.essential_task_list)}')
        total_distance = 0
        uav_num = 0
        while sum(self.essential_task_list) != 0:
            route_list = self.route_research(route_list=[], cur_route=Route(drop_off_num=len(self.drop_off_point_list)))
            sorted_route_list = sorted(route_list, key=lambda x: x.distance)
            selected_route = sorted_route_list[0]
            total_distance += selected_route.distance
            self.essential_task_list = [self.essential_task_list[idx] - selected_route.cargo_dist_list[idx] for idx in
                                        range(len(self.essential_task_list))]
            uav_num += 1
        print(f'总派出无人机次数： {uav_num}')
        print(f'总路程： {total_distance}')

    def route_research(self, route_list, cur_route):
        cur_distance = cur_route.distance
        cur_pos = cur_route.last_pos
        cur_cargo = sum(cur_route.cargo_dist_list)
        add_flag = True

        # 提前去掉没有任务的节点
        if cur_pos == -1:
            unvisited_nodes_copy = cur_route.unvisited_nodes.copy()
            for next_idx in unvisited_nodes_copy:
                if self.essential_task_list[next_idx] == 0:
                    cur_route.unvisited_nodes.remove(next_idx)

        # 判断能走的下一步，如果没有可达路径，则保存并返回
        # 如果没有满载，且能返航，则可以走下一步，否则保存返回
        if cur_cargo < UAV_MAX_CARGO_NUM:
            for next_idx in cur_route.unvisited_nodes:
                to_next_distance = self.distance_matrix[cur_pos + 1][next_idx + 1]
                next_back_distance = self.distance_matrix[next_idx + 1][0]
                add_distance = to_next_distance + next_back_distance
                if cur_distance + add_distance < UAV_MAX_DISTANCE:
                    add_flag = False
                    new_route = copy.deepcopy(cur_route)

                    new_route.route.append(next_idx)
                    add_cargo = min(self.essential_task_list[next_idx], UAV_MAX_CARGO_NUM - cur_cargo)
                    new_route.cargo_dist_list[next_idx] = add_cargo
                    new_route.distance += self.distance_matrix[cur_pos + 1][next_idx + 1]
                    new_route.unvisited_nodes.remove(next_idx)
                    new_route.last_pos = next_idx

                    self.route_research(route_list=route_list, cur_route=new_route)
        if add_flag:
            cur_route.distance += self.distance_matrix[cur_pos + 1][0]
            route_list.append(cur_route)
        return route_list


class Map:
    def __init__(self, width=30, height=30, dist_center_num=5, drop_off_point_num=25):
        self.width = width
        self.height = height
        self.dist_center_num = dist_center_num
        self.drop_off_point_num = drop_off_point_num
        self.dist_center_list = []
        self.drop_off_point_list = []
        self.invalid_drop_off_point_list = []
        self.create_map()

    def create_map(self):
        random.seed(123)
        point_num = self.dist_center_num + self.drop_off_point_num

        unique_points = set()
        while len(unique_points) < point_num:
            point = (random.uniform(0, self.width), random.uniform(0, self.height))
            unique_points.add(point)

        random_points = [[x, y] for (x, y) in unique_points]
        random.shuffle(random_points)
        self.dist_center_list = [DistCenterPoint(pos) for pos in random_points[:self.dist_center_num]]
        drop_off_point_tmp_list = random_points[self.dist_center_num:]

        # 将卸货点加入最近的配送中心；去掉超出极限距离的卸货点
        for point in drop_off_point_tmp_list:
            min_distance = 999
            cur_center_point = None
            for center_point in self.dist_center_list:
                distance = np.linalg.norm(np.array(point) - np.array(center_point.pos))
                if distance < min_distance:
                    min_distance = distance
                    cur_center_point = center_point
                elif distance == min_distance:
                    if len(cur_center_point.drop_off_point_list) < len(center_point.drop_off_point_list):
                        cur_center_point = center_point
            if min_distance > EXTREME_DISTANCE:
                self.invalid_drop_off_point_list.append(point)
            else:
                self.drop_off_point_list.append(point)
                cur_center_point.add_drop_off_point(point)

    def time_step_forward(self):
        """
        模拟每个时间刻
        :return:
        """
        for dist_center in self.dist_center_list:
            dist_center.update_unsolved_task_set()
            dist_center.generate_new_task()
            dist_center.route_optimize()


if __name__ == '__main__':
    cur_map = Map(width=WIDTH, height=HEIGHT, drop_off_point_num=DROP_OFF_POINT_NUM, dist_center_num=DIST_CENTER_NUM)
    plt.figure(figsize=(4.8, 4.8))
    plt.xlim(-3, cur_map.width + 3)
    plt.ylim(-3, cur_map.height + 3)
    dist_center_x = [dis_center.pos[0] for dis_center in cur_map.dist_center_list]
    dist_center_y = [dis_center.pos[1] for dis_center in cur_map.dist_center_list]
    plt.scatter(x=dist_center_x, y=dist_center_y, c='b', marker='o', label='distribution center')

    drop_off_point_x = [x for (x, y) in cur_map.drop_off_point_list]
    drop_off_point_y = [y for (x, y) in cur_map.drop_off_point_list]
    plt.scatter(x=drop_off_point_x, y=drop_off_point_y, c='g', marker='o', label='drop off point')

    invalid_drop_off_point_x = [x for (x, y) in cur_map.invalid_drop_off_point_list]
    invalid_drop_off_point_y = [y for (x, y) in cur_map.invalid_drop_off_point_list]
    plt.scatter(x=invalid_drop_off_point_x, y=invalid_drop_off_point_y, c='r', marker='o', label='unreachable point')
    # plt.legend(loc='upper right')
    plt.grid()
    plt.show()

    plt.figure(figsize=(4.8, 4.8))

    plt.xlim(-3, cur_map.width + 3)
    plt.ylim(-3, cur_map.height + 3)
    color_list = ['b', 'g', 'r', 'c', 'm']
    plt.grid()

    for c_i in range(len(color_list)):
        cur_center = cur_map.dist_center_list[c_i]
        cur_center_x = [point[0] for point in cur_center.drop_off_point_list]
        cur_center_y = [point[1] for point in cur_center.drop_off_point_list]
        plt.scatter(x=cur_center_x, y=cur_center_y, c=color_list[c_i], marker='o', label='drop off point')
        plt.scatter(x=cur_center.pos[0], y=cur_center.pos[1], c=color_list[c_i], marker='*', s=200,
                    label='distribution center')
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 0.5))
    # plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    for step_idx in range(STEP_NUM):
        print(f'第 {step_idx} 时间步')
        cur_map.time_step_forward()
