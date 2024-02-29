# Given an SU(d) matrix, analytically decompose it into SU(2) matrices

# https://arxiv.org/pdf/1708.00735.pdf

import numpy as np
from typing import Tuple

from bqskit import Circuit
from bqskit.ir.gates import EmbeddedGate
from bqskit.ir.gates import RYGate
from bqskit.ir.gates import U1Gate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import RXGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

def gen_rz_unitary(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0],
                     [0, np.exp(0.5j * theta)]])


def gen_ry_unitary(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]])


def gen_su2_unitary(alpha: float, beta: float, gamma: float) -> np.ndarray:
    return gen_rz_unitary(alpha) @ gen_ry_unitary(beta) @ gen_rz_unitary(gamma)


def gen_2mode_unitary(d: int, i: int, j: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Generate a 2-mode unitary from 3 Euler angles embedded in the i,j levels of a d radix unitary."""
    su2 = gen_su2_unitary(alpha, beta, gamma)
    embedded_su2 = np.identity(d, dtype=np.complex128)
    if i < j:
        assert i == j - 1
        embedded_su2[i:j+1, i:j+1] = su2
    else:
        assert j == i - 1
        embedded_su2[j:i+1, j:i+1] = su2
    return embedded_su2
    

def left_zero_ij_element(unitary: np.ndarray, i: int, j: int) -> Tuple[float, float, float]:
    """Return the params to an embedded U2 that will zero the i,j element of `unitary`."""
    if i < 0:
        i += unitary.shape[0]

    if j < 0:
        j += unitary.shape[0]

    assert i > j

    utry_col = unitary[:, j]
    w = utry_col[i]
    z = utry_col[i - 1]
    utry_col_without_w_and_z = np.concatenate([utry_col[:i-1], utry_col[i+1:]])
    if np.sum(np.square(np.abs(utry_col_without_w_and_z))) > 1:
        dem = 0
    else:
        dem = np.sqrt(1 - np.sum(np.square(np.abs(utry_col_without_w_and_z))))
    ws = w / dem
    zs = z / dem
    gamma = -(np.angle(zs) + np.angle(ws))
    alpha = -(np.angle(zs) - np.angle(ws))
    mag = np.abs(ws)
    if mag > 1:
        mag = 1
    beta = 2 * np.arcsin(mag)

    return alpha, beta, gamma


def right_zero_ij_element(unitary: np.ndarray, i: int, j: int) -> Tuple[float, float, float]:
    """Return the params to an embedded U2 that will zero the i,j element of `unitary`."""
    if i < 0:
        i += unitary.shape[0]

    if j < 0:
        j += unitary.shape[0]

    assert i > j

    utry_row = unitary[i, :]
    w = utry_row[j]
    z = utry_row[j + 1]
    utry_row_without_w_and_z = np.concatenate([utry_row[:j], utry_row[j+2:]])
    if np.sum(np.square(np.abs(utry_row_without_w_and_z))) > 1:
        dem = 0
    else:
        dem = np.sqrt(1 - np.sum(np.square(np.abs(utry_row_without_w_and_z))))
    ws = w / dem
    zs = z / dem
    gamma = -(np.angle(zs) + np.angle(-ws))
    alpha = -(np.angle(zs) - np.angle(-ws))
    mag = np.abs(zs)
    if mag > 1:
        mag = 1
    beta = 2 * np.arccos(mag)

    return alpha, beta, gamma


def build_circuit_column_by_column(unitary: UnitaryMatrix) -> Circuit:
    """Build a circuit that implements the given single-qudit unitary."""
    d = unitary.shape[0]
    circuit = Circuit(1, [d])
    target = unitary.to_special()

    for i in range(d - 1):
        for j in reversed(range(i + 1, d)):
            a, b, g = left_zero_ij_element(target, j, i)
            rz = EmbeddedGate(RZGate(), [d], [j - 1, j])
            ry = EmbeddedGate(RYGate(), [d], [j - 1, j])
            circuit.insert_gate(0, rz, 0, [a])
            circuit.insert_gate(0, ry, 0, [b])
            circuit.insert_gate(0, rz, 0, [g])
            eu = gen_2mode_unitary(d, j - 1, j, a, b, g)
            target = eu.conj().T @ target

    return circuit


def build_circuit_row_by_row(unitary: UnitaryMatrix) -> Circuit:
    """Build a circuit that implements the given single-qudit unitary."""
    d = unitary.shape[0]
    circuit = Circuit(1, [d])
    target = unitary.to_special()

    for i in reversed(range(1, d)):
        for j in range(i):

            if j != i - 1 and target[i, j] == 0:
                continue

            if j == i - 1 and target[i, j] == 0 and target[i, j + 1] == 1:
                continue

            a, b, g = right_zero_ij_element(target, i, j)
            rz = EmbeddedGate(RZGate(), [d], [j, j + 1])
            ry = EmbeddedGate(RYGate(), [d], [j, j + 1])
            circuit.append_gate(rz, 0, [-a])
            circuit.append_gate(ry, 0, [-b])
            circuit.append_gate(rz, 0, [-g])
            eu = gen_2mode_unitary(d, j, j + 1, a, b, g)
            target = target @ eu

    return circuit


def build_circuit_square(unitary: UnitaryMatrix) -> Circuit:
    """Build a circuit that implements the given single-qudit unitary."""
    d = unitary.shape[0]
    circuit = Circuit(1, [d])
    target = unitary.to_special()

    left_circuit = Circuit(1, [d])
    right_circuit = Circuit(1, [d])
    
    for diag in range(1, d):
        iterator = reversed(range(diag)) if diag % 2 == 1 else range(diag)
        for j in iterator:
            i = d - diag + j

            if j != i - 1 and target[i, j] == 0:
                continue

            if j == i - 1 and target[i, j] == 0 and target[i, j + 1] == 1:
                continue

            if diag % 2 == 1:
                a, b, g = right_zero_ij_element(target, i, j)
                rz = EmbeddedGate(RZGate(), [d], [j, j + 1])
                ry = EmbeddedGate(RYGate(), [d], [j, j + 1])
                left_circuit.append_gate(rz, 0, [-a])
                left_circuit.append_gate(ry, 0, [-b])
                left_circuit.append_gate(rz, 0, [-g])
                eu = gen_2mode_unitary(d, j, j + 1, a, b, g)
                target = target @ eu

            else:
                a, b, g = left_zero_ij_element(target, i, j)
                rz = EmbeddedGate(RZGate(), [d], [i - 1, i])
                ry = EmbeddedGate(RYGate(), [d], [i - 1, i])
                right_circuit.insert_gate(0, rz, 0, [a])
                right_circuit.insert_gate(0, ry, 0, [b])
                right_circuit.insert_gate(0, rz, 0, [g])
                eu = gen_2mode_unitary(d, i - 1, i, a, b, g)
                target = eu.conj().T @ target
    
    circuit.append_circuit(left_circuit, 0)
    circuit.append_circuit(right_circuit, 0)
    return circuit

def build_circuit_square_native(unitary: UnitaryMatrix) -> Circuit:
    """Build a circuit that implements the given single-qudit unitary."""
    d = unitary.shape[0]
    circuit = Circuit(1, [d])
    target = unitary.to_special()

    left_circuit = Circuit(1, [d])
    right_circuit = Circuit(1, [d])
    
    for diag in range(1, d):
        iterator = reversed(range(diag)) if diag % 2 == 1 else range(diag)
        for j in iterator:
            i = d - diag + j

            if False and target[i, j] < 1e-8:
                if j != i - 1:
                    continue

                if diag % 2 == 1 and target[i, j + 1] - 1 < 1e-8:
                    continue

                if diag % 2 != 1 and target[i + 1, j] - 1 < 1e-8:
                    continue

            if diag % 2 == 1:
                a, b, g = right_zero_ij_element(target, i, j)
                rz = EmbeddedGate(RZGate(), [d], [j, j + 1])
                ry = EmbeddedGate(RYGate(), [d], [j, j + 1])
                u1 = EmbeddedGate(U1Gate(), [d], [i - 1, i])
                rx = EmbeddedGate(RXGate().with_all_frozen_params([np.pi/2]), [d], [j, j + 1])
                left_circuit.append_gate(rz, 0, [-a])
                left_circuit.append_gate(rx, 0)
                left_circuit.append_gate(rz, 0, [-b + np.pi])
                left_circuit.append_gate(rx, 0)
                left_circuit.append_gate(rz, 0, [-g - np.pi])
                left_circuit.append_gate(u1, 0, [2 * np.pi])
                eu = gen_2mode_unitary(d, j, j + 1, a, b, g)
                target = target @ eu

            else:
                a, b, g = left_zero_ij_element(target, i, j)
                rz = EmbeddedGate(RZGate(), [d], [i - 1, i])
                u1 = EmbeddedGate(U1Gate(), [d], [i - 1, i])
                ry = EmbeddedGate(RYGate(), [d], [i - 1, i])
                rx = EmbeddedGate(RXGate().with_all_frozen_params([np.pi/2]), [d], [i - 1, i])

                right_circuit.insert_gate(0, u1, 0, [2 * np.pi])
                right_circuit.insert_gate(0, rz, 0, [a - np.pi])
                right_circuit.insert_gate(0, rx, 0)
                right_circuit.insert_gate(0, rz, 0, [b + np.pi])
                right_circuit.insert_gate(0, rx, 0)
                right_circuit.insert_gate(0, rz, 0, [g])
                eu = gen_2mode_unitary(d, i - 1, i, a, b, g)
                target = eu.conj().T @ target
    
    circuit.append_circuit(left_circuit, 0)
    circuit.append_circuit(right_circuit, 0)
    return circuit


def test_row_by_row(d):
    for _ in range(100):
        in_unitary = UnitaryMatrix.random(1, [d])
        circuit = build_circuit_row_by_row(in_unitary)
        assert circuit.get_unitary().get_distance_from(in_unitary) < 5e-7


def test_col_by_col(d):
    for _ in range(100):
        in_unitary = UnitaryMatrix.random(1, [d])
        circuit = build_circuit_column_by_column(in_unitary)
        assert circuit.get_unitary().get_distance_from(in_unitary) < 5e-7


def test_square(d):
    for _ in range(100):
        in_unitary = UnitaryMatrix.random(1, [d])
        circuit = build_circuit_square(in_unitary)
        assert circuit.get_unitary().get_distance_from(in_unitary) < 5e-7


def test_square_native(d):
    for _ in range(100):
        in_unitary = UnitaryMatrix.random(1, [d])
        circuit = build_circuit_square_native(in_unitary)
        assert circuit.get_unitary().get_distance_from(in_unitary) < 5e-7


if __name__ == "__main__":
    test_row_by_row(2)
    test_col_by_col(2)
    test_square(2)
    test_square_native(2)

    test_row_by_row(3)
    test_col_by_col(3)
    test_square(3)
    test_square_native(3)

    test_row_by_row(4)
    test_col_by_col(4)
    test_square(4)
    test_square_native(4)

    test_row_by_row(5)
    test_col_by_col(5)
    test_square(5)
    test_square_native(5)
