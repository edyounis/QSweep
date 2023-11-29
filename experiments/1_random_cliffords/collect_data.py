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

from qsearch import QSearchLayerGenQuquartSQ, QSearchLayerGenQutritSQ, QSearchLayerGenQubitSQ
sys.path.append('..')
from utils import count_pulses
from analytical import build_circuit_row_by_row, build_circuit_column_by_column


ds = [2, 3, 4, 5]
cliffords = {d: np.load(f'd={d}_cliffs.npy', allow_pickle=True)[:100] for d in ds}

# Compile with cbc analytical method
cbc_analytical_data = {
    d: {
        'times': [],
        'num_operations': [],
        'num_pulses': [],
        'circuits': [],
    }
    for d in ds
}

for d in ds:
    for i, utry in enumerate(cliffords[d]):
        if i % 10 == 0:
            print(f"Analytical CBC Compiling d = {d}, # {i}...")
        utry = UnitaryMatrix(utry, [utry.shape[0]])
        start = timer()
        circuit = build_circuit_column_by_column(utry)
        end = timer()
        cbc_analytical_data[d]['times'].append(end - start)
        cbc_analytical_data[d]['num_operations'].append(circuit.num_operations)
        cbc_analytical_data[d]['circuits'].append(circuit)
        cbc_analytical_data[d]['num_pulses'].append(count_pulses(circuit))


# Compile with rbr analytical method
rbr_analytical_data = {
    d: {
        'times': [],
        'num_operations': [],
        'num_pulses': [],
        'circuits': [],
    }
    for d in ds
}

for d in ds:
    for i, utry in enumerate(cliffords[d]):
        if i % 10 == 0:
            print(f"Analytical RBR Compiling d = {d}, # {i}...")
        utry = UnitaryMatrix(utry, [utry.shape[0]])
        start = timer()
        circuit = build_circuit_row_by_row(utry)
        end = timer()
        rbr_analytical_data[d]['times'].append(end - start)
        rbr_analytical_data[d]['num_operations'].append(circuit.num_operations)
        rbr_analytical_data[d]['circuits'].append(circuit)
        rbr_analytical_data[d]['num_pulses'].append(count_pulses(circuit))


# Compile with qsweep
qsweep_data = {
    d: {
        'times': [],
        'num_operations': [],
        'num_pulses': [],
        'circuits': [],
    }
    for d in ds
}

qsweep = QSweepPass({RZGate(), RXGate().with_all_frozen_params([np.pi/2])})

for d in ds:
    for i, utry in enumerate(cliffords[d]):
        if i % 10 == 0:
            print(f"QSweep Compiling d = {d}, # {i}...")
            utry = UnitaryMatrix(utry, [utry.shape[0]])
            start = timer()
            circuit = qsweep.synthesize_non_async(utry)
            end = timer()
            qsweep_data[d]['times'].append(end - start)
            qsweep_data[d]['num_operations'].append(circuit.num_operations)
            qsweep_data[d]['circuits'].append(circuit)
            qsweep_data[d]['num_pulses'].append(count_pulses(circuit))

# Compile with qsearch
qsearch_data = {
    d: {
        'times': [],
        'num_operations': [],
        'num_pulses': [],
        'circuits': [],
    }
    for d in ds
}

compiler = Compiler()

for d in ds[:2]:
    for i, utry in enumerate(cliffords[d]):
        if i % 10 == 0:
            print(f"QSearch Compiling d = {d}, # {i}...")
        utry = UnitaryMatrix(utry, [utry.shape[0]])
        circuit = Circuit.from_unitary(utry.to_special())
        start = timer()
        circuit = compiler.compile(circuit, [
            QSearchSynthesisPass(
                layer_generator=QSearchLayerGenQubitSQ() if d == 2 else QSearchLayerGenQutritSQ(),
                heuristic_function=DijkstraHeuristic(),
            )
        ])
        end = timer()
        qsearch_data[d]['times'].append(end - start)
        qsearch_data[d]['num_operations'].append(circuit.num_operations)
        qsearch_data[d]['circuits'].append(circuit)
        qsearch_data[d]['num_pulses'].append(count_pulses(circuit))

compiler.close()

import pickle
with open('experiment_data.pkl', 'wb') as f:
    pickle.dump({
        'cbc': cbc_analytical_data,
        'rbr': rbr_analytical_data,
        'qsweep': qsweep_data,
        'qsearch': qsearch_data,
    }, f)
