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

sys.path.append('..')
from utils import count_pulses
from analytical import build_circuit_row_by_row
from qsearch import QSearchLayerGenQuquartSQ, QSearchLayerGenQutritSQ

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

# Compile with analytical method
analytical_data = {
    'times': {},
    'num_operations': {},
    'num_pulses': {},
    'circuits': {},
}

for name, utry in unitaries.items():
    print(f"Compiling {name}...")
    utry = UnitaryMatrix(utry, [utry.shape[0]])
    start = timer()
    circuit = build_circuit_row_by_row(utry)
    end = timer()
    analytical_data['times'][name] = end - start
    analytical_data['num_operations'][name] = circuit.num_operations
    analytical_data['circuits'][name] = circuit
    analytical_data['num_pulses'][name] = count_pulses(circuit)

print("Times:", np.mean(analytical_data['times']), np.std(analytical_data['times']))
print("Operations:", np.mean(analytical_data['num_operations']), np.std(analytical_data['num_operations']))
print("Pulses:", np.mean(analytical_data['num_pulses']), np.std(analytical_data['num_pulses']))

# Compile with qsweep
qsweep_data = {
    'times': [],
    'num_operations': [],
    'num_pulses': [],
    'circuits': [],
}

qsweep = QSweepPass({RZGate(), RXGate().with_all_frozen_params([np.pi/2])})

for name, utry in unitaries.items():
    print(f"Compiling {name}...")
    utry = UnitaryMatrix(utry, [utry.shape[0]])
    start = timer()
    circuit = qsweep.synthesize_non_async(utry)
    end = timer()
    qsweep_data['times'][name] = end - start
    qsweep_data['num_operations'][name] = circuit.num_operations
    qsweep_data['circuits'][name] = circuit
    qsweep_data['num_pulses'][name] = count_pulses(circuit)

print("Times:", np.mean(qsweep_data['times']), np.std(qsweep_data['times']))
print("Operations:", np.mean(qsweep_data['num_operations']), np.std(qsweep_data['num_operations']))
print("Pulses:", np.mean(qsweep_data['num_pulses']), np.std(qsweep_data['num_pulses']))

# Compile with qsearch
qsearch_data = {
    'times': [],
    'num_operations': [],
    'num_pulses': [],
    'circuits': [],
}

compiler = Compiler()

for name, utry in unitaries.items():
    if name in ["qft", "xx", 'yy']:
        continue
    print(f"Compiling {name}...")
    utry = UnitaryMatrix(utry, [utry.shape[0]])
    circuit = Circuit.from_unitary(utry.to_special())
    start = timer()
    circuit = compiler.compile(circuit, [
        QSearchSynthesisPass(
            layer_generator=QSearchLayerGenQuquartSQ(),
            heuristic_function=DijkstraHeuristic(),
        )
    ])
    end = timer()
    qsearch_data['times'][name] = end - start
    qsearch_data['num_operations'][name] = circuit.num_operations
    qsearch_data['circuits'][name] = circuit
    qsearch_data['num_pulses'][name] = count_pulses(circuit)

compiler.close()
print("Times:", np.mean(qsearch_data['times']), np.std(qsearch_data['times']))
print("Operations:", np.mean(qsearch_data['num_operations']), np.std(qsearch_data['num_operations']))
print("Pulses:", np.mean(qsearch_data['num_pulses']), np.std(qsearch_data['num_pulses']))

import pickle
with open('experiment_data.pkl', 'wb') as f:
    pickle.dump({
        'analytical': analytical_data,
        'qsweep': qsweep_data,
        'qsearch': qsearch_data,
    }, f)
