import sys

import numpy as np

from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.parameterized.rx import RXGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.ir.circuit import Circuit
from bqskit.compiler import Compiler
from bqskit.passes import ForEachBlockPass, UnfoldPass
from qsweep import QSweepPass

sys.path.append('..')
from utils import format_to_qtrl

qsweep = QSweepPass({RZGate(), RXGate().with_all_frozen_params([np.pi/2])})

compiler = Compiler()

for d in ['qutrit', 'ququart']
    qtrl_circs = []
    circs = np.load(f'{d}_rb_circs_for_ed.npy', allow_pickle=True)
    for i, circ in enumerate(circs):

        print(f"On {d} circuit {i} out of {len(circs)}...")

        # Load dimension
        d = len(circ[0])

        # Construct circuit object
        circuit = Circuit(1, [d])
        for utry in circ:
            circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix(utry, [d]).to_special()), 0)

        # Compile
        start = timer()
        out_circuit = compiler.compile(circuit, [
            ForEachBlockPass([qsweep]),
            UnfoldPass(),
        ])
        end = timer()

        
        qtrl_circs.append(format_to_qtrl(out_circuit))
        
    import pickle
    with open(f'{d}_rb_circs_compiled_qtrl_for_noah.pkl', 'wb') as f:
        pickle.dump(qtrl_circs, f)

compiler.close()
