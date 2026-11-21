
import numpy as np
from src.mess_waste.drift import calculate_psi


def test_psi_identical_data():
    data = np.arange(100)
    assert calculate_psi(data, data) == 0.0


def test_psi_detects_large_shift():
    data = np.arange(100)
    shifted = data + 100

    psi = calculate_psi(data, shifted)

    assert psi > 0.0


def test_psi_detects_negative_shift():
    data = np.arange(100)
    shifted = data - 100

    psi = calculate_psi(data, shifted)

    assert psi > 0.0
