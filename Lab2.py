import cplex
import networkx as nx
from math import floor
from time import time
import os
import sys
from itertools import combinations


class TimeException(Exception):
    pass


def read_graph(path):
    text = open(path).read()
    list_edges = []
    for line in text.split("\n"):
        elems = line.split(" ")
        if elems[0] == "e":
            list_edges.append([int(elems[1]), int(elems[2])])
    return nx.Graph(list_edges)


class GraphSolver:
    def __init__(self, path, stop_time):
        self.graph = read_graph(path)
        self.dimension = len(self.graph.nodes)
        self.start_time = time()
        self.stop_time = stop_time
        self.timer = 0
        self.ind_sets = self.get_ind_sets()
        self.problem = self.set_problem()
        self.set_constraints()
        self.current_maximum_clique = 0
        self.branch_num = 0
        self.eps = 1e-6
        self.solution = []

    def update_timer(self):
        self.timer = time() - self.start_time

    def get_graph(self):
        return self.graph

    def solve(self):
        self.update_timer()
        if self.timer > self.stop_time:
            raise TimeException
        try:
            self.problem.solve()
            return self.problem.solution.get_values()
        except cplex.exceptions.CplexSolverError:
            return []

    def get_ind_sets(self):
        result = []
        strategies = [nx.coloring.strategy_largest_first,
                      nx.coloring.strategy_random_sequential,
                      nx.coloring.strategy_independent_set,
                      nx.coloring.strategy_connected_sequential_bfs,
                      nx.coloring.strategy_connected_sequential_dfs,
                      nx.coloring.strategy_saturation_largest_first,
                      nx.coloring.strategy_smallest_last,
                      nx.coloring.strategy_connected_sequential]

        for strategy in strategies:
            d = nx.coloring.greedy_color(self.graph, strategy=strategy)
            for color in set(color for _, color in d.items()):
                result.append(
                    [key for key, value in d.items() if value == color])
        return result

    def set_problem(self):
        problem = cplex.Cplex()
        problem.set_log_stream(None)
        problem.set_results_stream(None)
        problem.set_warning_stream(None)
        problem.set_error_stream(None)
        problem.objective.set_sense(problem.objective.sense.maximize)
        names = [f"x{i}" for i in self.graph.nodes]
        objective = [1.0] * self.dimension
        lower_bounds = [0.0] * self.dimension
        types = 'C' * self.dimension
        upper_bounds = [1.0] * self.dimension
        problem.variables.add(obj=objective,
                              lb=lower_bounds,
                              ub=upper_bounds,
                              names=names,
                              types=types)
        return problem

    def set_constraints(self):
        count_of_constraints = len(self.ind_sets)
        constraint_names = [f"c{i+1}" for i in range(count_of_constraints)]
        constraints = [[[f"x{i}" for i in elem], [1.0] * len(elem)] for elem in self.ind_sets]
        rhs = [1.0] * count_of_constraints
        constraint_senses = ["L"] * count_of_constraints

        self.problem.linear_constraints.add(lin_expr=constraints,
                                            senses=constraint_senses,
                                            rhs=rhs,
                                            names=constraint_names)

    def branch_and_cut(self):
        solution = self.solve()
        if len(solution) == 0:
            return
        if self.round_down(self.problem.solution.get_objective_value()) <= self.current_maximum_clique:
            return
        solution = self.increase_constraint(solution)
        if len(solution) == 0:
            return
        selected_var = self.get_branching_variable(solution)
        if selected_var is None:
            c = self.check_solution(solution)
            if len(c) != 0:
                self.add_constraint(c)
                self.branch_and_cut()
            else:
                new_solution_value = [index for index, value in enumerate(solution)
                                      if 1 - self.eps < value < 1 + self.eps]
                self.current_maximum_clique = len(new_solution_value)
                self.solution = solution
                print(f"Current max clique len: {self.current_maximum_clique}")
            return
        self.branching(selected_var)

    def branching(self, selected_var):
        branch_name1 = f"branch_{str(selected_var)}_{str(1.0)}"
        self.add_single_constrain(selected_var, 1.0, branch_name1)
        self.branch_and_cut()
        self.problem.linear_constraints.delete(branch_name1)
        branch_name2 = f"branch_{str(selected_var)}_{str(0.0)}"
        self.add_single_constrain(selected_var, 0.0, branch_name2)
        self.branch_and_cut()
        self.problem.linear_constraints.delete(branch_name2)

    def increase_constraint(self, solution):
        prev_f = 0
        repeat = 0
        internal_sol = solution
        while True:
            constrain = self.separation(internal_sol)
            if len(constrain) == 0:
                break
            self.add_constraint(constrain)
            internal_sol = self.solve()
            if len(internal_sol) == 0:
                break

            round_down = self.round_down(self.problem.solution.get_objective_value())
            if round_down <= self.current_maximum_clique:
                return []
            if round_down - prev_f < self.eps:
                repeat += 1
            prev_f = round_down
            if repeat > 1:
                break
        return internal_sol

    def round_down(self, value):
        rounded = floor(value)
        is_up = 1 if 1 - (value - rounded) <= self.eps else 0
        return rounded + is_up

    def get_branching_variable(self, solution):
        max_val = None
        index = None
        for i, val in enumerate(solution):
            is_zero = self.eps > val > 0 - self.eps
            is_one = 1 - self.eps < val < 1 + self.eps
            if not (is_zero or is_one):
                if max_val is None or max_val < val:
                    max_val = val
                    index = i
        return index

    def check_solution(self, solution):
        if self.is_clique(solution):
            return []
        else:
            return self.find_independent_set(solution)

    def find_independent_set(self, solution):
        index_nodes = [index for index, value in enumerate(solution) if value > self.eps]
        index_nodes = [x + 1 for x in index_nodes]
        ind_set = nx.maximal_independent_set(self.graph.subgraph(index_nodes))
        return nx.maximal_independent_set(self.graph, ind_set)

    def add_constraint(self, sets):
        name = "c"
        name = name + ",".join([str(x) for x in sets])
        self.problem.linear_constraints.add(
            lin_expr=[[[node_index - 1 for node_index in sets], [1.0] * len(sets)]],
            senses=['L'],
            rhs=[1.0],
            names=[name])

    def is_clique(self, solution):
        index_nodes = [index for index, value in enumerate(solution) if 1 - self.eps < value < 1 + self.eps]
        index_nodes = [x + 1 for x in index_nodes]
        for (u, v) in combinations(index_nodes, 2):
            if not self.graph.has_edge(u, v):
                return False
        return True

    def add_single_constrain(self, bv, rhs, name):
        self.problem.linear_constraints.add(
            lin_expr=[[[bv], [1.0]]],
            senses=['E'],
            rhs=[rhs],
            names=[name])

    def separation(self, solution):
        ind_sol = self.get_ind_first_max(solution)
        iter_val = 0
        while True:
            iter_val += 1
            if iter_val > 30:
                return []
            ind_set = nx.maximal_independent_set(self.graph, [ind_sol + 1])
            sum_val = 0
            for node in ind_set:
                sum_val += solution[node - 1]
            if sum_val > 1.0:
                return ind_set

    @staticmethod
    def get_ind_first_max(array):
        index, max_val = 0, 0
        for i, v in enumerate(array):
            if v > max_val:
                max_val = v
                index = i
        return index


def main():
    sys.setrecursionlimit(10000)
    for root, dirs, files in os.walk("files_test"):
        for filename in files:
            file = open("result2.txt", "a")
            print(f"Graph: {filename}")
            file.write(f"Graph: {filename}\n")
            start_time = time()
            solver = GraphSolver("files_test/" + filename, 7200)
            # graph = solver.get_graph()
            # print(graph.edges)
            try:
                solver.branch_and_cut()
                solution = solver.solution
                clique = solver.current_maximum_clique
            except TimeException:
                solution = solver.solution
                clique = solver.current_maximum_clique
                file.write("Execution failed on timer\n")

            work_time = round(time() - start_time, 1)
            file.write(f"Work time: {work_time} second\n")
            file.write(f"Maximum clique size: {clique}\n")
            print(solution)
            file.write(f"Nodes: {list(index + 1 for index, value in enumerate(solution) if value == 1.0)}\n")
            file.write("--------------------------------\n")
            file.close()


if __name__ == '__main__':
    main()
