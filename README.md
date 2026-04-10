# PDM4AR – Exercise 14: Multi-Agent Collection

Solutions for Exercise 14 of *Planning and Decision-Making for Autonomous Robots (PDM4AR)* at ETH Zurich.
This exercise considers multi-agent planning in a shared 2D environment, where several robots must collect assigned targets and deliver them to designated collection points while avoiding obstacles and mutual collisions.

Full problem description, evaluation details, and setup instructions are available on the [course website](https://pdm4ar.github.io/exercises/14-multiagent_collection.html).

---

## Repository Structure

The implementation follows the course template, with the main logic placed under `src/pdm4ar/exercises/`.

| Module | Responsibility |
|---|---|
| `planner.py` | Global multi-agent planning pipeline |
| `agent.py` | Agent interface and execution logic |
| `allocation.py` | Target assignment and ordering strategy |
| `path_planning.py` | Geometric path generation on the roadmap |

---

## Exercise Overview

The objective is to coordinate multiple robots in a common workspace so that each agent:

- Reaches assigned pickup targets
- Delivers collected items to the correct collection points
- Avoids static obstacles
- Avoids collisions with other agents during execution

The problem combines task allocation and motion planning. A good solution must produce efficient assignments while also generating feasible paths in a cluttered, shared environment.

---

## Planning Pipeline

The planner is organized as a sequence of stages:

1. Construct a probabilistic roadmap over the free space.
2. Connect valid samples with collision-free edges.
3. Query shortest paths between relevant task locations.
4. Assign targets to agents using a combinatorial optimization routine.
5. Refine multi-goal sequences to improve overall cost.
6. Plan agents sequentially while accounting for previously reserved space or trajectories.

This decomposition makes the full problem easier to solve while keeping the method scalable to larger instances.

---

## Implementation Notes

- **Roadmap-based planning:** the free space is represented through sampled nodes and valid local connections.
- **Task allocation:** assignment is handled separately from low-level path generation, which keeps the design modular.
- **Inter-agent safety:** agents are not planned independently; each path must remain compatible with the others.
- **Deterministic behavior:** fixed seeds and consistent planning order help make results reproducible.

---

## Running the Simulation

```bash
# Build and launch via Docker
make build
make run

# Or run with Poetry
poetry install
poetry run python -m pdm4ar.exercises_def.ex14.runner
```

The evaluation framework measures task completion, path feasibility, collision avoidance, and overall planning performance.
