import sys
import numpy as np
from timeit import default_timer as timer

from bqskit.qis.unitary import UnitaryMatrix
from bqskit.ir.gates import RXGate, RZGate
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes.search.heuristics.dijkstra import DijkstraHeuristic

from qsweep import QSweepPass

from experiments.utils import count_pulses

# Generating unitaries

qft_unitary = 0.5 * np.array([
    [1, 1, 1, 1],
    [1, 1j, -1, -1j],
    [1, -1, 1, -1],
    [1, -1j, -1, 1j],
])

from bqskit.ir.gates import CNOTGate, CZGate, CYGate, SwapGate, CHGate, XXGate, YYGate, ZZGate

unitaries = {"qft": qft_unitary}
for name, gates in [
    ('cx', CNOTGate()),
    ('cz', CZGate()),
    ('cy', CYGate()),
    ('swap', SwapGate()),
    ('ch', CHGate()),
    ('xx', XXGate()),
    ('yy', YYGate()),
    ('zz', ZZGate()),
]:
    unitaries[name] = UnitaryMatrix(gates.get_unitary().numpy, [4])

# Compile with qsweep
qsweep_data = {
    'times': {},
    'num_operations': {},
    'num_pulses': {},
    'circuits': {},
}

qsweep = QSweepPass({RZGate(), RXGate().with_all_frozen_params([np.pi/2])})

for name, utry in unitaries.items():
    print(f"Compiling {name}...")
    utry = UnitaryMatrix(utry, [utry.shape[0]])
    start = timer()
    circuit = qsweep.synthesize_non_async(utry)
    end = timer()
    print('times', end - start)
    print('num_operations', circuit.num_operations)
    print('num_pulses', count_pulses(circuit))
    print('error', utry.get_distance_from(circuit.get_unitary()))
    print()
    print()
