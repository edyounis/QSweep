import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING

from bqskit.ir.gates import EmbeddedGate
from bqskit.passes.search.generator import LayerGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.heuristics.dijkstra import DijkstraHeuristic
from bqskit import Circuit
from bqskit.compiler.passdata import PassData
from bqskit.ir.gate import Gate
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem

from bqskit.qis.unitary.unitary import RealVector
from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.opt.cost.function import CostFunction


class ElemsToZero(DifferentiableCostFunction):
    def __init__(self, circuit: Circuit, target: UnitaryMatrix, i_s: list[int], j_s: list[int], even: bool) -> None:
        """Initialize the SingleQuditLevelCost."""
        self.target = target
        self.target_dagger = target.conj().T
        self.circuit = circuit
        self.i_s = i_s
        self.j_s = j_s
        self.even = even

    def get_cost(self, params: RealVector) -> float:
        """Return the cost value given the input parameters."""
        utry = self.circuit.get_unitary(params)
        if self.even:
            inner_prod = utry.dagger @ self.target
        else:
            inner_prod = self.target_dagger @ utry
        cs = inner_prod[self.i_s, self.j_s]
        return np.sum(np.square(np.real(cs)) + np.square(np.imag(cs)))

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        """Return the residuals gradient given the input parameters."""
        return self.get_cost_and_grad(params)[1]
    
    def get_cost_and_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        """Return the residuals gradient given the input parameters."""
        utry, grads = self.circuit.get_unitary_and_grad(params)
        if self.even:
            inner_prod = utry.dagger @ self.target
        else:
            inner_prod = self.target_dagger @ utry
        cs = inner_prod[self.i_s, self.j_s]
        cost = np.sum(np.square(np.real(cs)) + np.square(np.imag(cs)))

        cost_grads = []
        for grad in grads:
            if self.even:
                inner_prod_grad = grad.conj().T @ self.target
            else:
                inner_prod_grad = self.target_dagger @ grad
            dcs = inner_prod_grad[self.i_s, self.j_s]
            cost_grads.append(np.sum(2 * (np.real(cs) * np.real(dcs) + np.imag(cs) * np.imag(dcs))))
        
        return cost, np.array(cost_grads)

class ElemsToZeroGenerator(CostFunctionGenerator):
    def __init__(self, i_s: list[int], j_s: list[int], even: bool = False) -> None:
        self.i_s = i_s
        self.j_s = j_s
        self.even = even

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
    ) -> CostFunction:
        """Generate a CostFunction, see CostFunctionGenerator for more info."""
        return ElemsToZero(circuit, target, self.i_s, self.j_s, self.even)


class ElemsToZeroAndOneToOneCost(DifferentiableCostFunction):
    def __init__(self, circuit: Circuit, target: UnitaryMatrix, i_s: list[int], j_s: list[int], ri: int, rj: int, even: bool) -> None:
        """Initialize the SingleQuditLevelCost."""
        self.target = target
        self.target_dagger = target.conj().T
        self.circuit = circuit
        self.i_s = i_s
        self.j_s = j_s
        self.ri = ri
        self.rj = rj
        self.even = even

    def get_cost(self, params: RealVector) -> float:
        """Return the cost value given the input parameters."""
        utry = self.circuit.get_unitary(params)
        if self.even:
            inner_prod = utry.dagger @ self.target
        else:
            inner_prod = self.target_dagger @ utry
        cs = inner_prod[self.i_s, self.j_s]
        zero_part_cost = np.sum(np.square(np.real(cs)) + np.square(np.imag(cs)))
        real_part_cost = (1 - inner_prod[self.ri, self.rj].real)**2 + np.square(np.imag(inner_prod[self.ri, self.rj]))
        return zero_part_cost + real_part_cost

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        """Return the residuals gradient given the input parameters."""
        return self.get_cost_and_grad(params)[1]
    
    def get_cost_and_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        """Return the residuals gradient given the input parameters."""
        utry, grads = self.circuit.get_unitary_and_grad(params)
        if self.even:
            inner_prod = utry.dagger @ self.target
        else:
            inner_prod = self.target_dagger @ utry
        cs = inner_prod[self.i_s, self.j_s]
        zero_part_cost = np.sum(np.square(np.real(cs)) + np.square(np.imag(cs)))
        real_part_cost = (1 - inner_prod[self.ri, self.rj].real)**2 + np.square(np.imag(inner_prod[self.ri, self.rj]))
        cost = zero_part_cost + real_part_cost

        cost_grads = []
        for grad in grads:
            if self.even:
                inner_prod_grad = grad.conj().T @ self.target
            else:
                inner_prod_grad = self.target_dagger @ grad
            dcs = inner_prod_grad[self.i_s, self.j_s]
            zero_part_grad = np.sum(2 * (np.real(cs) * np.real(dcs) + np.imag(cs) * np.imag(dcs)))
            real_part_grad = 2 * np.imag(inner_prod_grad[self.ri, self.rj]) * np.imag(inner_prod[self.ri, self.rj])
            real_part_grad += -2 * (1 - inner_prod[self.ri, self.rj].real) * inner_prod_grad[self.ri, self.rj].real
            cost_grads.append(real_part_grad + zero_part_grad)
        
        return cost, np.array(cost_grads)

class ElemsToZeroAndOneToOneCostGenerator(CostFunctionGenerator):
    def __init__(self, i_s: list[int], j_s: list[int], ri: int, rj: int, even: bool = False) -> None:
        self.i_s = i_s
        self.j_s = j_s
        self.ri = ri
        self.rj = rj
        self.even = even

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
    ) -> CostFunction:
        """Generate a CostFunction, see CostFunctionGenerator for more info."""
        return ElemsToZeroAndOneToOneCost(circuit, target, self.i_s, self.j_s, self.ri, self.rj, self.even)


class ElemsToZeroAndOneCost(DifferentiableCostFunction):
    def __init__(self, circuit: Circuit, target: UnitaryMatrix, zi_s: list[int], zj_s: list[int], ri_s: list[int], rj_s: list[int], even: bool) -> None:
        """Initialize the SingleQuditLevelCost."""
        self.target = target
        self.target_dagger = target.conj().T
        self.circuit = circuit
        self.zi_s = zi_s
        self.zj_s = zj_s
        self.ri_s = ri_s
        self.rj_s = rj_s
        self.even = even

    def get_cost(self, params: RealVector) -> float:
        """Return the cost value given the input parameters."""
        utry = self.circuit.get_unitary(params)
        if self.even:
            inner_prod = utry.dagger @ self.target
        else:
            inner_prod = self.target_dagger @ utry
        cs = inner_prod[self.zi_s, self.zj_s]
        zero_part_cost = np.sum(np.square(np.real(cs)) + np.square(np.imag(cs)))
        real_part_cost = np.sum((1 - inner_prod[self.ri_s, self.rj_s].real)**2 + np.square(np.imag(inner_prod[self.ri_s, self.rj_s])))
        return zero_part_cost + real_part_cost

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        """Return the residuals gradient given the input parameters."""
        return self.get_cost_and_grad(params)[1]
    
    def get_cost_and_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        """Return the residuals gradient given the input parameters."""
        utry, grads = self.circuit.get_unitary_and_grad(params)
        if self.even:
            inner_prod = utry.dagger @ self.target
        else:
            inner_prod = self.target_dagger @ utry
        cs = inner_prod[self.zi_s, self.zj_s]
        zero_part_cost = np.sum(np.square(np.real(cs)) + np.square(np.imag(cs)))
        real_part_cost = np.sum((1 - inner_prod[self.ri_s, self.rj_s].real)**2 + np.square(np.imag(inner_prod[self.ri_s, self.rj_s])))
        cost = zero_part_cost + real_part_cost

        cost_grads = []
        for grad in grads:
            if self.even:
                inner_prod_grad = grad.conj().T @ self.target
            else:
                inner_prod_grad = self.target_dagger @ grad
            dcs = inner_prod_grad[self.zi_s, self.zj_s]
            zero_part_grad = np.sum(2 * (np.real(cs) * np.real(dcs) + np.imag(cs) * np.imag(dcs)))
            real_part_grad = np.sum(2 * np.imag(inner_prod_grad[self.ri_s, self.rj_s]) * np.imag(inner_prod[self.ri_s, self.rj_s]))
            real_part_grad += np.sum(-2 * (1 - inner_prod[self.ri_s, self.rj_s].real) * inner_prod_grad[self.ri_s, self.rj_s].real)
            cost_grads.append(real_part_grad + zero_part_grad)
        
        return cost, np.array(cost_grads)

class ElemsToZeroAndOneCostGenerator(CostFunctionGenerator):
    def __init__(self, zi_s: list[int], zj_s: list[int], ri_s: list[int], rj_s: list[int], even: bool = False) -> None:
        self.zi_s = zi_s
        self.zj_s = zj_s
        self.ri_s = ri_s
        self.rj_s = rj_s
        self.even = even

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
    ) -> CostFunction:
        """Generate a CostFunction, see CostFunctionGenerator for more info."""
        return ElemsToZeroAndOneCost(circuit, target, self.zi_s, self.zj_s, self.ri_s, self.rj_s, self.even)
    

class ExactUnitaryCost(DifferentiableCostFunction):
    def __init__(self, circuit: Circuit, target: UnitaryMatrix) -> None:
        """Initialize the SingleQuditLevelCost."""
        self.target = target
        self.circuit = circuit

    def get_cost(self, params: RealVector) -> float:
        """Return the cost value given the input parameters."""
        utry = self.circuit.get_unitary(params)
        diff = self.target - utry
        return np.sum(np.square(np.real(diff))) + np.sum(np.square(np.imag(diff)))

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        """Return the residuals gradient given the input parameters."""
        return self.get_cost_and_grad(params)[1]
    
    def get_cost_and_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        """Return the residuals gradient given the input parameters."""
        utry, grads = self.circuit.get_unitary_and_grad(params)
        diff = self.target - utry
        cost = np.sum(np.square(np.real(diff))) + np.sum(np.square(np.imag(diff)))

        cost_grads = []
        for grad in grads:
            cost_grads.append(np.sum(-2*np.real(diff)*np.real(grad)) + np.sum(-2*np.imag(diff)*np.imag(grad)))
    
        return cost, np.array(cost_grads)

class ExactUnitaryCostGenerator(CostFunctionGenerator):
    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
    ) -> CostFunction:
        """Generate a CostFunction, see CostFunctionGenerator for more info."""
        return ExactUnitaryCost(circuit, target)


class QSweepLayerSimpleGen(LayerGenerator):
    def __init__(self, su2_gateset: set[Gate]) -> None:
        self.su2_gateset = su2_gateset
        self.cached_embedded_gates = {
            (2, 0, 1): su2_gateset,
        }

    def gen_initial_layer(self, target, data) -> Circuit:
        circuit = Circuit(target.num_qudits, target.radixes)
        return circuit

    def gen_successors(self, circuit: Circuit, data: PassData) -> list[Circuit]:
        i = data['qsweep_layer_gen_j']
        j = i + 1

        successors = []
        for gate in self.get_embedded_gates(circuit.radixes[0], i, j):
            if circuit.num_operations > 0 and circuit[0, 0].gate == gate:
                continue
            successor = circuit.copy()
            successor.insert_gate(0, gate, 0)
            successors.append(successor)
        return successors
    
    def get_embedded_gates(self, d: int, i: int, j: int) -> set[Gate]:
        if (d, i, j) in self.cached_embedded_gates:
            return self.cached_embedded_gates[(d, i, j)]
        else:
            embedded_gates = set()
            for gate in self.su2_gateset:
                embedded_gates.add(EmbeddedGate(gate, [d], [i, j]))
            self.cached_embedded_gates[(d, i, j)] = embedded_gates
            return embedded_gates
        
class QSweepLayerSquareGen(LayerGenerator):
    def __init__(self, su2_gateset: set[Gate]) -> None:
        self.su2_gateset = su2_gateset
        self.cached_embedded_gates = {
            (2, 0, 1): su2_gateset,
        }

    def gen_initial_layer(self, target, data) -> Circuit:
        circuit = Circuit(target.num_qudits, target.radixes)
        return circuit

    def gen_successors(self, circuit: Circuit, data: PassData) -> list[Circuit]:
        i = data['qsweep_layer_gen_j']
        j = i + 1

        successors = []
        for gate in self.get_embedded_gates(circuit.radixes[0], i, j):
            if circuit.num_operations > 0 and circuit[0, 0].gate == gate:
                    continue
            successor = circuit.copy()
            successor.insert_gate(0, gate, 0)
            successors.append(successor)
        return successors
    
    def get_embedded_gates(self, d: int, i: int, j: int) -> set[Gate]:
        if (d, i, j) in self.cached_embedded_gates:
            return self.cached_embedded_gates[(d, i, j)]
        else:
            embedded_gates = set()
            for gate in self.su2_gateset:
                embedded_gates.add(EmbeddedGate(gate, [d], [i, j]))
            self.cached_embedded_gates[(d, i, j)] = embedded_gates
            return embedded_gates


class QSweepPass(SynthesisPass):
    def __init__(self, su2_gate_set: set[Gate]) -> None:
        self.su2_gate_set = su2_gate_set

    async def synthesize(
        self,
        utry: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        if not isinstance(utry, UnitaryMatrix):
            raise RuntimeError('QSweepPass only supports unitary matrices.')
        
        if utry.num_qudits != 1:
            raise RuntimeError('QSweepPass only supports single qudit synthesis.')
        
        layer_gen = QSweepLayerSimpleGen(self.su2_gate_set)

        target = utry.to_special()
        circuit = Circuit(1, utry.radixes)

        for d in reversed(range(1, utry.radixes[0])):
            subcircuit = layer_gen.gen_initial_layer(target, data)

            if d >= 2:

                for i in range(d):
                    data['qsweep_layer_gen_j'] = i
                    subcircuit = self.set_elem_to_zero(target, subcircuit, layer_gen, i, d, data)

                assert(np.abs(1 - (target.dagger @ subcircuit.get_unitary())[-1, -1].real) < 1e-7)
                target = UnitaryMatrix((target.dagger @ subcircuit.get_unitary())[:-1, :-1], (target.radixes[0]-1,)).dagger
            
            else:
                subcircuit = self.synthesize_su2(target, subcircuit, layer_gen, data)

            circuit.insert_circuit(0, self.elevate_subcircuit(subcircuit, utry.radixes[0]), 0)
        
        return circuit
    
    def synthesize_non_async(self, utry: UnitaryMatrix) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        if not isinstance(utry, UnitaryMatrix):
            raise RuntimeError('QSweepPass only supports unitary matrices.')
        
        if utry.num_qudits != 1:
            raise RuntimeError('QSweepPass only supports single qudit synthesis.')
        
        layer_gen = QSweepLayerSimpleGen(self.su2_gate_set)

        target = utry.to_special()
        circuit = Circuit(1, utry.radixes)
        data = {}

        for d in reversed(range(1, utry.radixes[0])):
            subcircuit = layer_gen.gen_initial_layer(target, data)

            if d >= 2:
                for i in range(d):
                    subcircuit = self.set_elem_to_zero(target, subcircuit, layer_gen, i, d, data)

                assert(np.abs(1 - (target.dagger @ subcircuit.get_unitary())[-1, -1].real) < 1e-7)
                target = UnitaryMatrix((target.dagger @ subcircuit.get_unitary())[:-1, :-1], (target.radixes[0]-1,)).dagger
            
            else:
                subcircuit = self.synthesize_su2(target, subcircuit, layer_gen, data)

            circuit.insert_circuit(0, self.elevate_subcircuit(subcircuit, utry.radixes[0]), 0)
        
        return circuit
    
    def elevate_subcircuit(self, subcircuit, i):
        """subcircuit is radix j, elevate to radix i"""
        if subcircuit.radixes[0] == i:
            return subcircuit

        circuit = Circuit(1, [i])

        assert i > subcircuit.radixes[0]

        for op in subcircuit:
            if isinstance(op.gate, EmbeddedGate):
                lmap = op.gate.level_maps[0]
                gate = op.gate.gate
                circuit.append_gate(EmbeddedGate(gate, [i], lmap), 0, op.params)
            else:
                circuit.append_gate(EmbeddedGate(op.gate, [i], [0, 1]), 0, op.params)
        
        return circuit
    
    def success_critera(self, target, circuit, i, d) -> bool:
        remainder = target.dagger @ circuit.get_unitary()
        is_zero = np.abs(remainder[-1, i]) < 1e-7

        if i == d - 1:
            is_one = np.abs(1 - remainder[-1, -1].real) < 1e-7
            return is_zero and is_one
        
        return is_zero
        

    def set_elem_to_zero(self, target, circuit, layer_gen, i, d, data):
        if self.success_critera(target, circuit, i, d):
            return circuit
        
        frontier = Frontier(target, DijkstraHeuristic())

        cost = ElemsToZeroGenerator([-1] * (i + 1), [j for j in range(i + 1)])
        if i == d - 1:
            cost = ElemsToZeroAndOneToOneCostGenerator([-1] * (i + 1), [j for j in range(i + 1)], -1, -1)    
        
        circuit.instantiate(
            target,
            method="minimization",
            cost_fn_gen=cost,
            minimizer=LBFGSMinimizer(),
            multistarts=1,
        )
        
        if self.success_critera(target, circuit, i, d):
            return circuit

        frontier.add(circuit, 0)
        data['qsweep_layer_gen_j'] = i

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
                if self.success_critera(target, circuit, i, d):
                    return circuit

                frontier.add(circuit, layer + 1)
        
        raise RuntimeError("Frontier emptied.")

    def synthesize_su2(self, target, circuit, layer_gen, data):
        data['qsweep_layer_gen_j'] = 0
        if ExactUnitaryCostGenerator().calc_cost(circuit, target) < 5e-8:
            return circuit
        
        frontier = Frontier(target, DijkstraHeuristic())

        circuit.instantiate(
            target,
            method="minimization",
            cost_fn_gen=ExactUnitaryCostGenerator(),
            minimizer=LBFGSMinimizer(),
            multistarts=1,
        )

        if ExactUnitaryCostGenerator().calc_cost(circuit, target) < 5e-8:
            return circuit

        frontier.add(circuit, 0)

        while not frontier.empty():
            top_circuit, layer = frontier.pop()

            successors = layer_gen.gen_successors(top_circuit, data)

            if len(successors) == 0:
                continue

            circuits = [
                s.instantiate(
                    target,
                    method="minimization",
                    cost_fn_gen=ExactUnitaryCostGenerator(),
                    minimizer=LBFGSMinimizer(),
                    multistarts=1,
                )
                for s in successors
            ]

            for circuit in circuits:
                if ExactUnitaryCostGenerator().calc_cost(circuit, target) < 5e-8:
                    return circuit

                frontier.add(circuit, layer + 1)
        
        raise RuntimeError("Frontier emptied.")
