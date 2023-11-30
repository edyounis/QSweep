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

import pickle
with open('experiment_data.pkl', 'rb') as f:
    data = pickle.load(f)

for d in ds:
    for i, utry in enumerate(cliffords[d]):
        in_utry = UnitaryMatrix(utry, [utry.shape[0]])
        out_utry = data['rbr'][d]['circuits'][i].get_unitary()
        assert in_utry.get_distance_from(out_utry) < 5e-8

for d in ds:
    for i, utry in enumerate(cliffords[d]):
        in_utry = UnitaryMatrix(utry, [utry.shape[0]])
        out_utry = data['qsweep'][d]['circuits'][i].get_unitary()
        assert in_utry.get_distance_from(out_utry) < 5e-8

for d in ds[:2]:
    for i, utry in enumerate(cliffords[d]):
        in_utry = UnitaryMatrix(utry, [utry.shape[0]])
        out_utry = data['qsearch'][d]['circuits'][i].get_unitary()
        # Note: QSearch is not exact, so we have to be a bit more lenient
        assert in_utry.get_distance_from(out_utry) < 2e-4
