import numpy as np
from bqskit.ir.gates.composed.embedded import EmbeddedGate
from bqskit.ir.gates.constant.sx import SqrtXGate
from bqskit.ir.gates.parameterized.rx import RXGate
from bqskit.ir.gates.parameterized.ry import RYGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.circuit import Circuit


X90 = RXGate().with_all_frozen_params([np.pi/2])
X90_3 = EmbeddedGate(X90, [3], [0, 1])
EFX90_3 = EmbeddedGate(X90, [3], [1, 2])
X90_4 = EmbeddedGate(X90, [4], [0, 1])
EFX90_4 = EmbeddedGate(X90, [4], [1, 2])
FHX90_4 = EmbeddedGate(X90, [4], [2, 3])


Z = RZGate()
Z_3 = EmbeddedGate(Z, [3], [0, 1])
Z_4 = EmbeddedGate(Z, [4], [0, 1])
EFZ_3 = EmbeddedGate(Z, [3], [1, 2])
EFZ_4 = EmbeddedGate(Z, [4], [1, 2])
FHZ_4 = EmbeddedGate(Z, [4], [2, 3])


def count_pulses(circuit: Circuit) -> int:
    """
    Count the number of pulses a single-qudit circuit.

    Note:
        This is designed for superconducting quantum machines that implement
        a virtual-z gate.
    """
    count = 0
    for op in circuit:
        if isinstance(op.gate, EmbeddedGate):
            if op.gate.gate in [X90, SqrtXGate()]:
                count += 1
            elif op.gate.gate in [RYGate()]:
                count += 2
    return count


def format_to_qtrl(circuit: Circuit) -> list[list[str]]:
    qtrl_circ = []

    for op in circuit:
        gate = op.gate
        if gate == X90_3 or gate == X90_4:
            qtrl_circ.append(['Q4/X90'])
        
        elif gate == EFX90_3 or gate == EFX90_4:
            qtrl_circ.append(['Q4/EFX90'])

        elif gate == FHX90_4:
            qtrl_circ.append(['Q4/FHX90'])
        
        elif gate == Z_3:
            # np.exp(-1j*theta/2)*Z(theta)@EFZ(-theta/2)
            qtrl_circ.append([f'Q4/Z{op.params[0] * 180 / np.pi}'])
            qtrl_circ.append([f'Q4/EFZ{(-op.params[0]/2) * 180 / np.pi}'])

        elif gate == Z_4:
            # np.exp(-1j*theta/2)*Z(theta)@EFZ(-theta/2)
            qtrl_circ.append([f'Q4/Z{op.params[0] * 180 / np.pi}'])
            qtrl_circ.append([f'Q4/EFZ{(-op.params[0]/2) * 180 / np.pi}'])
        
        elif gate == EFZ_3:
            # Z(-theta/2)*EFZ(theta)
            qtrl_circ.append([f'Q4/Z{(-op.params[0]/2) * 180 / np.pi}'])
            qtrl_circ.append([f'Q4/EFZ{op.params[0] * 180 / np.pi}'])

        elif gate == EFZ_4:
            # Z(-theta/2)*EFZ(theta)@FHZ(-theta/2)
            qtrl_circ.append([f'Q4/Z{(-op.params[0]/2) * 180 / np.pi}'])
            qtrl_circ.append([f'Q4/EFZ{op.params[0] * 180 / np.pi}'])
            qtrl_circ.append([f'Q4/FHZ{(-op.params[0]/2) * 180 / np.pi}'])
        
        elif gate == FHZ_4:
            # EFZ(-theta/2)@FHZ(theta)
            qtrl_circ.append([f'Q4/EFZ{(-op.params[0]/2) * 180 / np.pi}'])
            qtrl_circ.append([f'Q4/FHZ{op.params[0] * 180 / np.pi}'])

        else:
            raise NotImplementedError
        
    return qtrl_circ


def test_format_to_qtrl():
    Z = lambda theta : np.diag([1,np.exp(1j*theta),np.exp(1j*theta)])
    EFZ = lambda theta : np.diag([1,1,np.exp(1j*theta)])

    for theta in [0, 1, 2, 3, 4, 5, 2.23128917]:
        assert np.allclose(np.exp(-1j*theta/2)*Z(theta)@EFZ(-theta/2), Z_3.get_unitary([theta]))
        assert np.allclose(Z(-theta/2)*EFZ(theta), EFZ_3.get_unitary([theta]))

    Z = lambda theta : np.diag([1,np.exp(1j*theta),np.exp(1j*theta),np.exp(1j*theta)])
    EFZ = lambda theta : np.diag([1,1,np.exp(1j*theta),np.exp(1j*theta)])
    FHZ = lambda theta : np.diag([1,1,1,np.exp(1j*theta)])

    for theta in [0, 1, 2, 3, 4, 5, 2.23128917]:
        assert np.allclose(np.exp(-1j*theta/2)*Z(theta)@EFZ(-theta/2), Z_4.get_unitary([theta]))
        assert np.allclose(Z(-theta/2)*EFZ(theta)@FHZ(-theta/2), EFZ_4.get_unitary([theta]))
        assert np.allclose(EFZ(-theta/2)@FHZ(theta), FHZ_4.get_unitary([theta]))


if __name__ == "__main__":
    test_format_to_qtrl()