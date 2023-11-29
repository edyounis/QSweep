from bqskit.ir.gates import SqrtXGate, RZGate, EmbeddedGate, RYGate
from bqskit.passes import LayerGenerator
from bqskit.compiler import PassData
from bqskit.ir.circuit import Circuit


X90_01_3 = EmbeddedGate(SqrtXGate(), [3], [0, 1])
X90_12_3 = EmbeddedGate(SqrtXGate(), [3], [1, 2])
X90_01_4 = EmbeddedGate(SqrtXGate(), [4], [0, 1])
X90_12_4 = EmbeddedGate(SqrtXGate(), [4], [1, 2])
X90_23_4 = EmbeddedGate(SqrtXGate(), [4], [2, 3])

Z_01_3 = EmbeddedGate(RZGate(), [3], [0, 1])
Z_12_3 = EmbeddedGate(RZGate(), [3], [1, 2])
Z_01_4 = EmbeddedGate(RZGate(), [4], [0, 1])
Z_12_4 = EmbeddedGate(RZGate(), [4], [1, 2])
Z_23_4 = EmbeddedGate(RZGate(), [4], [2, 3])

class QSearchLayerGenQubitSQ(LayerGenerator):
    def gen_initial_layer(self, target, data) -> Circuit:
        circuit = Circuit(target.num_qudits, target.radixes)
        return circuit

    def gen_successors(self, circuit: Circuit, data: PassData) -> list[Circuit]:
        successors = []
        for gate in [SqrtXGate(), RZGate()]:
            if circuit.num_operations > 0 and circuit[-1, 0].gate == gate:
                continue
            successor = circuit.copy()
            successor.append_gate(gate, 0)
            successors.append(successor)
        return successors

class QSearchLayerGenQutritSQ(LayerGenerator):
    def gen_initial_layer(self, target, data) -> Circuit:
        circuit = Circuit(target.num_qudits, target.radixes)
        return circuit

    def gen_successors(self, circuit: Circuit, data: PassData) -> list[Circuit]:
        successors = []
        for gate in [X90_01_3, X90_12_3, Z_01_3, Z_12_3]:
            if circuit.num_operations > 0 and circuit[-1, 0].gate == gate:
                continue
            successor = circuit.copy()
            successor.append_gate(gate, 0)
            successors.append(successor)
        return successors


class QSearchLayerGenQuquartSQ(LayerGenerator):
    def gen_initial_layer(self, target, data) -> Circuit:
        circuit = Circuit(target.num_qudits, target.radixes)
        return circuit

    def gen_successors(self, circuit: Circuit, data: PassData) -> list[Circuit]:
        successors = []
        for gate in [X90_01_4, X90_12_4, X90_23_4, Z_01_4, Z_12_4, Z_23_4]:
            # if circuit.num_operations > 0 and circuit[-1, 0].gate == gate:
            #    continue  # This actually makes it slower :shrug:
            successor = circuit.copy()
            successor.append_gate(gate, 0)
            successors.append(successor)
        return successors
