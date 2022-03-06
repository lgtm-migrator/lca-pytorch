import unittest

import torch
from torch.testing import assert_close

from lcapt.preproc import standardize_inputs


class TestPreprocessing(unittest.TestCase):

    def test_standardize_inputs_raises_NotImplementedError(self):
        with self.assertRaises(NotImplementedError):
            standardize_inputs(torch.zeros(1, 2, 3, 4, 5, 6))

    def test_standardize_inputs_3D_tensor(self):
        inputs = torch.rand(1, 3, 100) * 10
        outputs = standardize_inputs(inputs)
        for chann in range(3):
            assert_close(outputs[:, chann].mean().item(), 0.0)
            assert_close(outputs[:, chann].std().item(), 1.0)

    def test_standardize_inputs_4D_tensor(self):
        inputs = torch.rand(1, 3, 100, 100) * 10
        outputs = standardize_inputs(inputs)
        for chann in range(3):
            assert_close(outputs[:, chann].mean().item(), 0.0)
            assert_close(outputs[:, chann].std().item(), 1.0)

    def test_standardize_inputs_5D_tensor(self):
        inputs = torch.rand(1, 3, 100, 100, 100) * 10
        outputs = standardize_inputs(inputs)
        for chann in range(3):
            assert_close(outputs[:, chann].mean().item(), 0.0, rtol=5e-4,
                         atol=5e-5)
            assert_close(outputs[:, chann].std().item(), 1.0, rtol=5e-4,
                         atol=5e-5)

    def test_standardize_inputs_already_standardized(self):
        inputs_randn = torch.randn(1, 3, 100, 100, 100)
        inputs_randn = standardize_inputs(inputs_randn)
        outputs = standardize_inputs(inputs_randn)
        assert_close(inputs_randn, outputs, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()