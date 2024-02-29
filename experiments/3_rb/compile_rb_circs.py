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
from test2 import QsweepSquarePass, SquareAnalyticalPass

# qsweep = QsweepSquarePass({RZGate(), RXGate().with_all_frozen_params([np.pi/2])})
qsweep = SquareAnalyticalPass()

compiler = Compiler()

# for dtype in ['qutrit', 'ququart']:
for dtype in ['ququart']:
    qtrl_circs = []
    circs = np.load(f'{dtype}_rb_circs_for_ed.npy', allow_pickle=True)
    for i, circ in enumerate(circs):

        print(f"On {dtype} circuit {i} out of {len(circs)}...")

        # Load dimension
        d = len(circ[0])

        # Construct circuit object
        circuit = Circuit(1, [d])
        for utry in circ:
            circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix(utry, [d]).to_special()), 0)

        # Compile
        out_circuit = compiler.compile(circuit, [
            ForEachBlockPass([qsweep]),
            UnfoldPass(),
        ])

        
        qtrl_circs.append(format_to_qtrl(out_circuit))
        
    import pickle
    with open(f'{dtype}_rb_circs_compiled_qtrl_for_noah_square_analytical.pkl', 'wb') as f:
        pickle.dump(qtrl_circs, f)

compiler.close()
