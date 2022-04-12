import cplex
import os
import networkx as nx
import time


class TimeException(Exception):
    pass


def read_graph(path):
    text = open(path).read()
    list_edges = []
    dimension = 0
    for line in text.split("\n"):
        elems = line.split(" ")
        if elems[0] == "e":
            list_edges.append([int(elems[1]), int(elems[2])])
        elif elems[0] == "p":
            dimension = int(elems[2])
    return nx.Graph(list_edges), dimension


class GraphSolver:
    def __init__(self, path, stop_time):
        self.graph, self.dimension = read_graph(path)
        self.start_time = time.time()
        self.stop_time = stop_time
        self.timer = time.time() - self.start_time
        self.not_connected = list(nx.complement(self.graph).edges)
        self.ind_sets = self.get_greedy_coloring()
        self.problem = self.set_problem()
        self.set_constraints()
        self.current_maximum_clique = 0
        self.branch_num = 0
        self.eps = 1e-6
        self.solution = 0

    def get_graph(self):
        return self.graph

    def solve(self):
        return self.branching()

    def check_stop_time(self):
        if self.timer > self.stop_time:
            raise TimeException

    def get_greedy_coloring(self):
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
            coloring = nx.coloring.greedy_color(self.graph, strategy=strategy)
            for color in set(color for node, color in coloring.items()):
                result.append([key for key, value in coloring.items() if value == color])

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
        upper_bounds = [cplex.infinity] * self.dimension
        problem.variables.add(obj=objective,
                              lb=lower_bounds,
                              ub=upper_bounds,
                              names=names,
                              types=types)
        return problem

    def set_constraints(self):
        constraint_names = [f"c{i}" for i in range((len(self.ind_sets) + len(self.not_connected)))]
        constraints = [[[f"x{i}" for i in elem], [1.0] * len(elem)] for elem in self.ind_sets + self.not_connected]
        rhs = [1.0] * (len(self.ind_sets) + len(self.not_connected))
        constraint_senses = ["L"] * (len(self.ind_sets) + len(self.not_connected))

        self.problem.linear_constraints.add(lin_expr=constraints,
                                            senses=constraint_senses,
                                            rhs=rhs,
                                            names=constraint_names)

    def add_constraint(self, bv, rhs, current_branch):
        self.problem.linear_constraints.add(lin_expr=[[[bv], [1]]],
                                            senses=['E'],
                                            rhs=[rhs],
                                            names=[f'branch_{current_branch}'])

    @staticmethod
    def get_branching_variable(solution):
        return max(list(filter(lambda x: not x[1].is_integer(), enumerate(solution))),
                   key=lambda x: x[1], default=(None, None))[0]

    def fix_solution(self, solution):
        if isinstance(solution, list):
            for i in range(len(solution)):
                if abs(solution[i] - 1.0) < self.eps:
                    solution[i] = 1.0
                elif abs(solution[i]) < self.eps:
                    solution[i] = 0.0
        return solution

    def branching(self):
        try:
            self.problem.solve()
            solution = self.problem.solution.get_values()
        except cplex.exceptions.CplexSolverError:
            return 0

        if sum(solution) > self.current_maximum_clique:
            solution = self.fix_solution(solution)
            self.timer = time.time() - self.start_time
            self.check_stop_time()
            branching_variable = self.get_branching_variable(solution)
            if branching_variable is None:
                self.current_maximum_clique = sum(solution)
                print(f"Max clique {self.current_maximum_clique} found after {round(self.timer, 1)} second")
                self.solution = solution
                return self.current_maximum_clique, solution
            else:
                self.branch_num += 1
                current_branch = self.branch_num
                self.add_constraint(branching_variable, 1.0, current_branch)
                branch_left = self.branching()
                self.problem.linear_constraints.delete(f'branch_{current_branch}')
                self.add_constraint(branching_variable, 0.0, current_branch)
                branch_right = self.branching()
                self.problem.linear_constraints.delete(f'branch_{current_branch}')

                return max([branch_left, branch_right], key=lambda x: x[0] if isinstance(x, (list, tuple)) else x)
        return 0


def check_clique(solution, graph):
    nodes = list(index for index, value in enumerate(solution) if value == 1.0)
    flag = True
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if not (nodes[i], nodes[j]) in graph.edges and not (nodes[i], nodes[j]) in graph.edges:
                flag = False
    return flag


def main():
    for root, dirs, files in os.walk("files_test"):
        for filename in files:
            file = open("result1_second.txt", "a")
            print(f"Graph: {filename}")
            file.write(f"Graph: {filename}\n")
            start_time = time.time()
            solver = GraphSolver("files_test/" + filename, 7200)
            # graph = solver.get_graph()
            # print(graph.edges)
            try:
                clique, solution = solver.solve()
            except TimeException:
                solution = solver.solution
                clique = solver.current_maximum_clique
                file.write("Execution failed on timer\n")

            work_time = round(time.time() - start_time, 1)
            file.write(f"Work time: {work_time} second\n")
            file.write(f"Maximum clique size: {clique}\n")
            print(solution)
            file.write(f"Nodes: {list(index + 1 for index, value in enumerate(solution) if value == 1.0)}\n")
            file.write("--------------------------------\n")
            file.close()


if __name__ == '__main__':
    main()
