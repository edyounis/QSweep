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

from qsearch import QSearchLayerGenQuquartSQ, QSearchLayerGenQutritSQ
sys.path.append('..')
from utils import count_pulses
from analytical import build_circuit_row_by_row, build_circuit_column_by_column

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

import pickle
with open('experiment_data.pkl', 'rb') as f:
    data = pickle.load(f)

for name, utry in unitaries.items():
    in_utry = UnitaryMatrix(utry, [utry.shape[0]])
    out_utry = data['rbr']['circuits'][name].get_unitary()
    assert in_utry.get_distance_from(out_utry) < 5e-8

for name, utry in unitaries.items():
    in_utry = UnitaryMatrix(utry, [utry.shape[0]])
    out_utry = data['qsweep']['circuits'][name].get_unitary()
    assert in_utry.get_distance_from(out_utry) < 5e-8

for name, utry in unitaries.items():
    if name not in data['qsearch']['circuits']:
        continue
    in_utry = UnitaryMatrix(utry, [utry.shape[0]])
    out_utry = data['qsearch']['circuits'][name].get_unitary()
    # Note: QSearch is not exact, so we have to be a bit more lenient
    assert in_utry.get_distance_from(out_utry) < 2e-4
