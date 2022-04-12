import cplex

# чтение файла
file = open("C125.9.clq.txt")
text = file.read()
# получение списка ребер
list_edges = []
for line in text.split("\n"):
    elems = line.split(" ")
    if elems[0] == "e":
        list_edges.append([int(elems[1]), int(elems[2])])
max_vertex = max(list_edges[-1])
# получение списка не ребер
list_not_edges = [[j, i] for i in range(1, max_vertex+1) for j in range(1, max_vertex+1) if i != j]
for edges in list_edges:
    list_not_edges.remove(edges)

# составление условий
list_constraints = [[[f"x{edges[0]}", f"x{edges[1]}"], [1.0, 1.0]] for edges in list_not_edges]

# список результатов
list_result = [1.0 for i in range(len(list_constraints))]
constraint_senses = ["L" for i in range(len(list_constraints))]
constraint_names = [f"c{i}" for i in range(len(list_constraints))]

# задание cplex
problem = cplex.Cplex()
problem.objective.set_sense(problem.objective.sense.maximize)
objective = [1.0 for i in range(max_vertex)]
names = [f"x{i+1}" for i in range(max_vertex)]
lower_bounds = [0.0 for i in range(max_vertex)]
upper_bounds = [cplex.infinity for i in range(max_vertex)]
problem.variables.add(obj=objective,
                      lb=lower_bounds,
                      ub=upper_bounds,
                      names=names)

# решение
problem.linear_constraints.add(lin_expr=list_constraints,
                               senses=constraint_senses,
                               rhs=list_result,
                               names=constraint_names)
problem.solve()
print("Решение в вещественных числах")
print(problem.solution.get_values())

