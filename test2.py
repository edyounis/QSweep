import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.ir.gates import RZGate, RXGate, EmbeddedGate
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.heuristics.dijkstra import DijkstraHeuristic
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer

from qsweep import QSweepLayerSquareGen, ElemsToZeroGenerator, ElemsToZeroAndOneToOneCostGenerator, ElemsToZeroAndOneCostGenerator



def get_cost(i, j, d):
    diag = d + j - i
    even = diag % 2 != 1

    zi_s = []
    zj_s = []
    for prev_diag in range(1, diag):
        if prev_diag % 2 == 1:
            for prev_j in reversed(range(prev_diag)):
                prev_i = d - prev_diag + prev_j
                zi_s.append(prev_i)
                zj_s.append(prev_j)
        else:
            for prev_j in range(prev_diag):
                prev_i = d - prev_diag + prev_j
                zi_s.append(prev_i)
                zj_s.append(prev_j)
    
    if not even:
        for prev_j in reversed(range(j, diag)):
            prev_i = d - diag + prev_j
            zi_s.append(prev_i)
            zj_s.append(prev_j)
    else:
        for prev_j in range(j + 1):
            prev_i = d - diag + prev_j
            zi_s.append(prev_i)
            zj_s.append(prev_j)
    
    ri_s = []
    rj_s = []
    if diag == d - 1:
        if not even:
            for diag_i in reversed(range(j + 1, d)):
                ri_s.append(diag_i)
                rj_s.append(diag_i)
            
            if j == 0:
                ri_s.append(0)
                rj_s.append(0)

        else:
            for diag_i in range(i):
                ri_s.append(diag_i)
                rj_s.append(diag_i)
            
            if i == d - 1:
                ri_s.append(d - 1)
                rj_s.append(d - 1)

    return ElemsToZeroAndOneCostGenerator(zi_s, zj_s, ri_s, rj_s, even)


def set_elem_to_zero(target, circuit, layer_gen, i, j, d, data):
        frontier = Frontier(target, DijkstraHeuristic())
        cost = get_cost(i, j, d)

        if cost.calc_cost(circuit, target) < 5e-8:
            return circuit
        
        frontier.add(circuit, 0)

        while not frontier.empty():
            top_circuit, layer = frontier.pop()

            successors = layer_gen.gen_successors(top_circuit, data)

            circuits = [
                s.instantiate(
                    target,
                    method="minimization",
                    cost_fn_gen=cost,
                    minimizer=LBFGSMinimizer(),
                    multistarts=1,
                )
                for s in successors
            ]

            for circuit in circuits:
                if cost.calc_cost(circuit, target) < 5e-8:
                    return circuit

                frontier.add(circuit, layer + 1)
        
        raise RuntimeError("Frontier emptied.")


def embed_circuit(circuit, factor_levels, d):
    embedded_circuit = Circuit(1, [d])
    for op in circuit:
        embedded_circuit.append_gate(
            EmbeddedGate(op.gate, [d], [factor_levels]),
            0,
            op.params,
        )
    return embedded_circuit


def calculate_inverse_factor(target, factor_levels, layer_gen, d):
    cost = ElemsToZeroAndOneCostGenerator([1], [0], [0, 1], [0, 1], False)
    frontier = Frontier(target, DijkstraHeuristic())
    data = {}
    data['qsweep_layer_gen_j'] = 0
    circuit = Circuit(1, [2])

    if cost.calc_cost(circuit, target) < 5e-8:
        return embed_circuit(circuit, factor_levels, d)
    
    frontier.add(circuit, 0)

    while not frontier.empty():
        top_circuit, layer = frontier.pop()

        successors = layer_gen.gen_successors(top_circuit, data)

        circuits = [
            s.instantiate(
                target,
                method="minimization",
                cost_fn_gen=cost,
                minimizer=LBFGSMinimizer(),
                multistarts=1,
            )
            for s in successors
        ]

        for circuit in circuits:
            if cost.calc_cost(circuit, target) < 5e-8:
                return embed_circuit(circuit, factor_levels, d)

            frontier.add(circuit, layer + 1)
        
    raise RuntimeError("Frontier emptied.")


def split_into_su2s_and_levels(subcircuit):
    su2_and_levels = []
    current_utry = None
    current_level = None

    for op in subcircuit:
        assert isinstance(op.gate, EmbeddedGate)

        if current_level is None or current_level != op.gate.level_maps[0]:
            if current_level is not None:
                su2_and_levels.append((current_utry, current_level))
            current_utry = UnitaryMatrix.identity(2)
            current_level = op.gate.level_maps[0]
        
        current_utry = op.gate.gate.get_unitary(op.params) @ current_utry
    
    if current_level is not None:
        su2_and_levels.append((current_utry, current_level))
    
    return su2_and_levels


def calculate_inverse(subcircuit, layer_gen, d):
    inverse_subcircuit = Circuit(1, [d])
    for su2, factor_levels in reversed(split_into_su2s_and_levels(subcircuit)):
        factor = calculate_inverse_factor(su2.dagger, factor_levels, layer_gen, d)
        inverse_subcircuit.append_circuit(factor, 0)
    return inverse_subcircuit


from bqskit.ir.gate import Gate

def qsweep_square(input_utry: UnitaryMatrix, su2_gate_set: set[Gate]):
    assert input_utry.num_qudits == 1
    d = input_utry.radixes[0]
    target = input_utry.to_special()
    circuit = Circuit(1, [d])
    gen = QSweepLayerSquareGen(su2_gate_set)
    data = {}
    center = 0


    for diag in range(1, d):
        even = diag % 2 != 1
        iterator = reversed(range(diag)) if not even else range(diag)
        subcircuit = Circuit(1, [d])

        for j in iterator:
            i = d - diag + j
            data['qsweep_layer_gen_j'] = (i - 1) if even else j
            subcircuit = set_elem_to_zero(target, subcircuit, gen, i, j, d, data)
        
        if not even:
            circuit.insert_circuit(center, subcircuit, 0)

        else:
            inverse_subcircuit = calculate_inverse(subcircuit, gen, d)
            circuit.insert_circuit(center, inverse_subcircuit, 0)
            center += len(subcircuit)
        
        target = target.dagger @ subcircuit.get_unitary()
    
    print(circuit.get_unitary().get_distance_from(input_utry))
    print(np.round(circuit.get_unitary(), 3))
    print()
    print(np.round(input_utry, 3))
    print()
    print(np.round(input_utry.dagger @ circuit.get_unitary(), 3))
    for op in circuit:
        print(op)
    assert circuit.get_unitary().get_distance_from(input_utry) < 5e-8
    return circuit


from bqskit.passes.synthesis.synthesis import SynthesisPass

class QsweepSquarePass(SynthesisPass):
    def __init__(self, su2_gate_set: set[Gate]) -> None:
        self.su2_gate_set = su2_gate_set

    async def synthesize(self, utry, data) -> Circuit:
        if not isinstance(utry, UnitaryMatrix):
            raise RuntimeError('QSweepPass only supports unitary matrices.')
        
        if utry.num_qudits != 1:
            raise RuntimeError('QSweepPass only supports single qudit synthesis.')
        
        print(utry)
        return qsweep_square(utry, self.su2_gate_set)


# in_utry = [[-2.77555756e-17+3.06161700e-17j,-3.06161700e-17+5.55111512e-17j,
#   -5.00000000e-01+5.00000000e-01j,  5.00000000e-01-5.00000000e-01j,],
#  [-5.00000000e-01-5.00000000e-01j, -5.00000000e-01-5.00000000e-01j,
#    3.06161700e-17-5.55111512e-17j,  2.77555756e-17+3.06161700e-17j],
#  [-5.55111512e-17+3.06161700e-17j, -3.06161700e-17+2.77555756e-17j,
#   -5.00000000e-01-5.00000000e-01j, -5.00000000e-01-5.00000000e-01j],
#  [-5.00000000e-01+5.00000000e-01j,  5.00000000e-01-5.00000000e-01j,
#    3.06161700e-17-5.55111512e-17j,  2.77555756e-17+3.06161700e-17j]]
# in_utry = UnitaryMatrix(in_utry, [4])
# qsweep_square(in_utry, {RZGate(), RXGate().with_all_frozen_params([np.pi/2])})

from analytical import build_circuit_square_native

class SquareAnalyticalPass(SynthesisPass):
    async def synthesize(self, utry, data) -> Circuit:
        if not isinstance(utry, UnitaryMatrix):
            raise RuntimeError('QSweepPass only supports unitary matrices.')
        
        if utry.num_qudits != 1:
            raise RuntimeError('QSweepPass only supports single qudit synthesis.')
        
        circ = build_circuit_square_native(utry)
        assert circ.get_unitary().get_distance_from(utry) < 5e-8
        return circ