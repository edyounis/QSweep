import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.ir.gates import RZGate, RXGate, EmbeddedGate
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.heuristics.dijkstra import DijkstraHeuristic
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer

from qsweep import QSweepLayerSquareGen, ElemsToZeroGenerator, ElemsToZeroAndOneToOneCostGenerator, ElemsToZeroAndOneCostGenerator

qft_unitary = UnitaryMatrix(0.5 * np.array([
    [1, 1, 1, 1],
    [1, 1j, -1, -1j],
    [1, -1, 1, -1],
    [1, -1j, -1, 1j],
]), [4])

in_utry = [[-2.77555756e-17+3.06161700e-17j,-3.06161700e-17+5.55111512e-17j,
  -5.00000000e-01+5.00000000e-01j,  5.00000000e-01-5.00000000e-01j,],
 [-5.00000000e-01-5.00000000e-01j, -5.00000000e-01-5.00000000e-01j,
   3.06161700e-17-5.55111512e-17j,  2.77555756e-17+3.06161700e-17j],
 [-5.55111512e-17+3.06161700e-17j, -3.06161700e-17+2.77555756e-17j,
  -5.00000000e-01-5.00000000e-01j, -5.00000000e-01-5.00000000e-01j],
 [-5.00000000e-01+5.00000000e-01j,  5.00000000e-01-5.00000000e-01j,
   3.06161700e-17-5.55111512e-17j,  2.77555756e-17+3.06161700e-17j]]
qft_unitary = UnitaryMatrix(in_utry, [4])

d = 4
target = qft_unitary.to_special()
circuit = Circuit(1, [d])
center = 0


subcircuit = Circuit(1, [d])

i = 3
j = 0
rz = EmbeddedGate(RZGate(), [d], [j, j + 1])
rx = EmbeddedGate(RXGate().with_all_frozen_params([np.pi/2]), [d], [j, j + 1])
subcircuit.insert_gate(0, rz, 0)
subcircuit.insert_gate(0, rx, 0)
subcircuit.insert_gate(0, rz, 0)
subcircuit.insert_gate(0, rx, 0)
subcircuit.insert_gate(0, rz, 0)

cost = ElemsToZeroGenerator([3], [0])
subcircuit.instantiate(
    target,
    method="minimization",
    cost_fn_gen=cost,
    minimizer=LBFGSMinimizer(),
    multistarts=1,
)
remainder = target.dagger @ subcircuit.get_unitary()
print(np.round(remainder, 3)); print()


# #############################################k
print("Next diag")
circuit.insert_circuit(center, subcircuit, 0)
center += 0
target = target.dagger @ subcircuit.get_unitary()
subcircuit = Circuit(1, [d])
print()
# #############################################

i = 2
j = 0
rz = EmbeddedGate(RZGate(), [d], [i - 1, i])
rx = EmbeddedGate(RXGate().with_all_frozen_params([np.pi/2]), [d], [i - 1, i])
subcircuit.insert_gate(0, rz, 0)
subcircuit.insert_gate(0, rx, 0)
subcircuit.insert_gate(0, rz, 0)
subcircuit.insert_gate(0, rx, 0)
subcircuit.insert_gate(0, rz, 0)

cost = ElemsToZeroGenerator([3, 2], [0, 0], True)
subcircuit.instantiate(
    target,
    method="minimization",
    cost_fn_gen=cost,
    minimizer=LBFGSMinimizer(),
    multistarts=1,
)
remainder = subcircuit.get_unitary().dagger @ target
print(np.round(remainder, 3)); print()


i = 3
j = 1
rz = EmbeddedGate(RZGate(), [d], [i - 1, i])
rx = EmbeddedGate(RXGate().with_all_frozen_params([np.pi/2]), [d], [i - 1, i])
subcircuit.insert_gate(0, rz, 0)
subcircuit.insert_gate(0, rx, 0)
subcircuit.insert_gate(0, rz, 0)
subcircuit.insert_gate(0, rx, 0)
subcircuit.insert_gate(0, rz, 0)

cost = ElemsToZeroGenerator([3, 2, 3], [0, 0, 1], True)
subcircuit.instantiate(
    target,
    method="minimization",
    cost_fn_gen=cost,
    minimizer=LBFGSMinimizer(),
    multistarts=1,
)
remainder = subcircuit.get_unitary().dagger @ target
print(np.round(remainder, 3)); print()


# #############################################k
print("Next diag")
for op in subcircuit:
    circuit.insert(center, op.get_inverse())
# circuit.insert_circuit(center, subcircuit, 0)
center += len(subcircuit)
target = target.dagger @ subcircuit.get_unitary()
subcircuit = Circuit(1, [d])
print()
# #############################################

i = 3
j = 2
rz = EmbeddedGate(RZGate(), [d], [j, j + 1])
rx = EmbeddedGate(RXGate().with_all_frozen_params([np.pi/2]), [d], [j, j + 1])
subcircuit.insert_gate(0, rz, 0)
subcircuit.insert_gate(0, rx, 0)
subcircuit.insert_gate(0, rz, 0)
subcircuit.insert_gate(0, rx, 0)
subcircuit.insert_gate(0, rz, 0)

cost = ElemsToZeroAndOneCostGenerator([3, 2, 3, 3], [0, 0, 1, 2], [3], [3], False)
subcircuit.instantiate(
    target,
    method="minimization",
    cost_fn_gen=cost,
    minimizer=LBFGSMinimizer(),
    multistarts=1,
)
remainder = target.dagger @ subcircuit.get_unitary()
print(np.round(remainder, 3)); print()

i = 2
j = 1
rz = EmbeddedGate(RZGate(), [d], [j, j + 1])
rx = EmbeddedGate(RXGate().with_all_frozen_params([np.pi/2]), [d], [j, j + 1])
subcircuit.insert_gate(0, rz, 0)
subcircuit.insert_gate(0, rx, 0)
subcircuit.insert_gate(0, rz, 0)
subcircuit.insert_gate(0, rx, 0)
subcircuit.insert_gate(0, rz, 0)

cost = ElemsToZeroAndOneCostGenerator([3, 2, 3, 3, 2], [0, 0, 1, 2, 1], [3, 2], [3, 2], False)
subcircuit.instantiate(
    target,
    method="minimization",
    cost_fn_gen=cost,
    minimizer=LBFGSMinimizer(),
    multistarts=1,
)
remainder = target.dagger @ subcircuit.get_unitary()
print(np.round(remainder, 3)); print()

i = 1
j = 0
rz = EmbeddedGate(RZGate(), [d], [j, j + 1])
rx = EmbeddedGate(RXGate().with_all_frozen_params([np.pi/2]), [d], [j, j + 1])
subcircuit.insert_gate(0, rz, 0)
subcircuit.insert_gate(0, rx, 0)
subcircuit.insert_gate(0, rz, 0)
subcircuit.insert_gate(0, rx, 0)
subcircuit.insert_gate(0, rz, 0)

cost = ElemsToZeroAndOneCostGenerator([3, 2, 3, 3, 2, 1], [0, 0, 1, 2, 1, 0], [3, 2, 1, 0], [3, 2, 1, 0], False)
subcircuit.instantiate(
    target,
    method="minimization",
    cost_fn_gen=cost,
    minimizer=LBFGSMinimizer(),
    multistarts=1,
)
remainder = target.dagger @ subcircuit.get_unitary()
print(np.round(remainder, 3)); print()

# #############################################k
print("Next diag")
# for op in subcircuit:
#     circuit.insert(center, op)
circuit.insert_circuit(center, subcircuit, 0)
center += 0
target = target.dagger @ subcircuit.get_unitary()
subcircuit = Circuit(1, [d])
print()
# #############################################

# for op in circuit:
#     print(op)
print(circuit.get_unitary().get_distance_from(qft_unitary))

d = 4
target = qft_unitary.to_special()
circuit = Circuit(1, [d])
gen = QSweepLayerSquareGen({RZGate(), RXGate().with_all_frozen_params([np.pi/2])})