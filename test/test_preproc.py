import unittest

import torch
from torch.testing import assert_close

from lcapt.preproc import contrast_norm, zero_mean


class TestPreprocessing(unittest.TestCase):
    def test_zero_mean_3D_tensor(self):
        inputs = torch.rand(10, 3, 1000) * 100
        outputs = zero_mean(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        for inp in range(10):
            self.assertLess(outputs[inp].mean().abs().item(), 1e-5)

    def test_contrast_norm_3D_tensor(self):
        inputs = torch.rand(10, 3, 1000) * 100
        outputs = contrast_norm(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        for inp in range(10):
            assert_close(outputs[inp].std().item(), 1.0)

    def test_zero_mean_4D_tensor(self):
        inputs = torch.rand(10, 3, 100, 100) * 100
        outputs = zero_mean(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        for inp in range(10):
            self.assertLess(outputs[inp].mean().abs().item(), 1e-5)

    def test_contrast_norm_4D_tensor(self):
        inputs = torch.rand(10, 3, 100, 100) * 100
        outputs = contrast_norm(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        for inp in range(10):
            assert_close(outputs[inp].std().item(), 1.0)

    def test_zero_mean_5D_tensor(self):
        inputs = torch.rand(10, 3, 4, 100, 100) * 100
        outputs = zero_mean(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        for inp in range(10):
            self.assertLess(outputs[inp].mean().abs().item(), 1e-5)

    def test_contrast_norm_5D_tensor(self):
        inputs = torch.rand(10, 3, 4, 100, 100) * 100
        outputs = contrast_norm(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        for inp in range(10):
            assert_close(outputs[inp].std().item(), 1.0)

    def test_zero_mean_inputs_already_zero_mean(self):
        inputs = torch.randn(10, 3, 10000)
        inputs = inputs - inputs.mean((-2, -1), True)
        outputs = zero_mean(inputs)
        assert_close(inputs, outputs)

    def test_contrast_norm_inputs_already_unit_var(self):
        inputs = torch.randn(10, 3, 10000)
        inputs = inputs / inputs.std((-2, -1), keepdim=True)
        outputs = contrast_norm(inputs)
        assert_close(inputs, outputs)

    def test_zero_mean_returns_torch_tensor(self):
        inputs = torch.rand(1, 3, 4, 100, 100)
        outputs = zero_mean(inputs)
        self.assertEqual(type(outputs), torch.Tensor)

    def test_contrast_norm_returns_torch_tensor(self):
        inputs = torch.rand(1, 3, 4, 100, 100)
        outputs = contrast_norm(inputs)
        self.assertEqual(type(outputs), torch.Tensor)


if __name__ == "__main__":
    unittest.main()
