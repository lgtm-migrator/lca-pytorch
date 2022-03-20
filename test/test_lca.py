from tempfile import TemporaryDirectory
import unittest

import torch
from torch.testing import assert_close

from lcapt.lca import LCAConv1D, LCAConv2D, LCAConv3D


class TestLCA(unittest.TestCase):
    def test_LCAConv1D_to_correct_input_shape_raises_ValueError(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir)
            inputs_4d = torch.zeros(1, 3, 10, 10)

            with self.assertRaises(ValueError):
                lca._to_correct_input_shape(inputs_4d)

    def test_LCAConv2D_to_correct_input_shape_raises_ValueError(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir)
            inputs_3d = torch.zeros(1, 3, 10)

            with self.assertRaises(ValueError):
                lca._to_correct_input_shape(inputs_3d)

    def test_LCAConv3D_to_correct_input_shape_raises_ValueError(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir)
            inputs_4d = torch.zeros(1, 3, 10, 10)

            with self.assertRaises(ValueError):
                lca._to_correct_input_shape(inputs_4d)

    def test_LCAConv1D_get_weights_returns_correct_shape(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, kt=5)
            weights = lca.get_weights()
            self.assertTrue(len(weights.shape) == 3)
            self.assertTupleEqual(weights.numpy().shape, (10, 3, 5))

    def test_LCAConv2D_get_weights_returns_correct_shape(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 5, 7)
            weights = lca.get_weights()
            self.assertTrue(len(weights.shape) == 4)
            self.assertTupleEqual(weights.numpy().shape, (10, 3, 5, 7))

    def test_LCAConv3D_get_weights_returns_correct_shape(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 5, 7, 9)
            weights = lca.get_weights()
            self.assertTrue(len(weights.shape) == 5)
            self.assertTupleEqual(weights.numpy().shape, (10, 3, 9, 5, 7))

    def test_LCAConv1D_assign_weight_values(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, kt=5)
            new_weights = torch.randn(10, 3, 5)
            lca.assign_weight_values(new_weights)
            assert_close(new_weights, lca.get_weights(), rtol=0, atol=0)

    def test_LCAConv2D_assign_weight_values(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 5, 7)
            new_weights = torch.randn(10, 3, 5, 7)
            lca.assign_weight_values(new_weights)
            assert_close(new_weights, lca.get_weights(), rtol=0, atol=0)

    def test_LCAConv3D_assign_weight_values(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 5, 7, 9)
            new_weights = torch.randn(10, 3, 9, 5, 7)
            lca.assign_weight_values(new_weights)
            assert_close(new_weights, lca.get_weights(), rtol=0, atol=0)

    def test_LCAConv1D_normalize_weights(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, kt=5)
            new_weights = torch.rand(10, 3, 5) * 10 + 10
            lca.assign_weight_values(new_weights)
            lca.normalize_weights()
            for feat in lca.get_weights():
                self.assertAlmostEqual(feat.norm(2).item(), 1.0, 5)

    def test_LCAConv2D_normalize_weights(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 5, 7)
            new_weights = torch.rand(10, 3, 5, 7) * 10 + 10
            lca.assign_weight_values(new_weights)
            lca.normalize_weights()
            for feat in lca.get_weights():
                self.assertAlmostEqual(feat.norm(2).item(), 1.0, 5)

    def test_LCAConv3D_normalize_weights(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 5, 7, 9)
            new_weights = torch.rand(10, 3, 9, 5, 7) * 10 + 10
            lca.assign_weight_values(new_weights)
            lca.normalize_weights()
            for feat in lca.get_weights():
                self.assertAlmostEqual(feat.norm(2).item(), 1.0, 5)

    def test_LCAConv1D_initial_weights_are_normalized(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, kt=5)
            for feat in lca.get_weights():
                self.assertAlmostEqual(feat.norm(2).item(), 1.0, 5)

    def test_LCAConv2D_initial_weights_are_normalized(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 5, 7)
            for feat in lca.get_weights():
                self.assertAlmostEqual(feat.norm(2).item(), 1.0, 5)

    def test_LCAConv3D_initial_weights_are_normalized(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 5, 7, 9)
            for feat in lca.get_weights():
                self.assertAlmostEqual(feat.norm(2).item(), 1.0, 5)

    def test_compute_input_pad_raises_ValueError(self):
        with TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                lca = LCAConv2D(10, 3, tmp_dir, 5, 7, pad="weird_padding")

    def test_LCAConv1D_input_padding_shape_same_padding(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, kt=5)
            self.assertTupleEqual(lca.input_pad, (2, 0, 0))

    def test_LCAConv2D_input_padding_shape_same_padding(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 5, 7)
            self.assertTupleEqual(lca.input_pad, (0, 2, 3))

    def test_LCAConv3D_input_padding_shape_same_padding(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 5, 7, 9)
            self.assertTupleEqual(lca.input_pad, (4, 2, 3))

    def test_LCAConv1D_input_padding_shape_valid_padding(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, kt=5, pad="valid")
            self.assertTupleEqual(lca.input_pad, (0, 0, 0))

    def test_LCAConv2D_input_padding_shape_valid_padding(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 5, 7, pad="valid")
            self.assertTupleEqual(lca.input_pad, (0, 0, 0))

    def test_LCAConv3D_input_padding_shape_valid_padding(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 5, 7, 9, pad="valid")
            self.assertTupleEqual(lca.input_pad, (0, 0, 0))

    def test_LCAConv1D_code_shape_stride_1_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, kt=5, lca_iters=3)
            inputs = torch.randn(1, 3, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 100))

    def test_LCAConv2D_code_shape_stride_1_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 5, 7, lca_iters=3)
            inputs = torch.randn(1, 3, 100, 99)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 100, 99))

    def test_LCAConv3D_code_shape_stride_1_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 5, 7, 9, lca_iters=3)
            inputs = torch.randn(1, 3, 8, 100, 101)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 8, 100, 101))

    def test_LCAConv1D_code_shape_stride_2_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, kt=5, lca_iters=3, stride_t=2)
            inputs = torch.randn(1, 3, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 50))

    def test_LCAConv2D_code_shape_stride_2_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 5, 7, lca_iters=3, stride_h=2, stride_w=2)
            inputs = torch.randn(1, 3, 100, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 50, 50))

    def test_LCAConv3D_code_shape_stride_2_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10, 3, tmp_dir, 5, 7, 9, lca_iters=3, stride_h=2, stride_w=2, stride_t=2
            )
            inputs = torch.randn(1, 3, 8, 100, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 4, 50, 50))

    def test_LCAConv1D_code_shape_stride_4_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, kt=5, lca_iters=3, stride_t=4)
            inputs = torch.randn(1, 3, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 25))

    def test_LCAConv2D_code_shape_stride_4_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 5, 7, lca_iters=3, stride_h=4, stride_w=4)
            inputs = torch.randn(1, 3, 100, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 25, 25))

    def test_LCAConv3D_code_shape_stride_4_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10, 3, tmp_dir, 5, 7, 9, lca_iters=3, stride_h=4, stride_w=4, stride_t=4
            )
            inputs = torch.randn(1, 3, 8, 100, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 2, 25, 25))

    def test_LCAConv1D_recon_shape_stride_1_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, kt=5, lca_iters=3, return_all=True)
            inputs = torch.randn(1, 3, 100)
            _, recon, _, _, _ = lca(inputs)
            assert_close(recon.shape, inputs.shape, rtol=0, atol=0)

    def test_LCAConv2D_recon_shape_stride_1_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 5, 7, lca_iters=3, return_all=True)
            inputs = torch.randn(1, 3, 100, 100)
            _, recon, _, _, _ = lca(inputs)
            assert_close(inputs.shape, recon.shape, rtol=0, atol=0)

    def test_LCAConv3D_recon_shape_stride_1_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 5, 7, 9, lca_iters=3, return_all=True)
            inputs = torch.randn(1, 3, 8, 100, 100)
            _, recon, _, _, _ = lca(inputs)
            assert_close(inputs.shape, recon.shape, rtol=0, atol=0)

    def test_LCAConv1D_recon_shape_stride_2_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5, 2, lca_iters=3, return_all=True)
            inputs = torch.randn(1, 3, 100)
            _, recon, _, _, _ = lca(inputs)
            assert_close(recon.shape, inputs.shape, rtol=0, atol=0)

    def test_LCAConv2D_recon_shape_stride_2_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 5, 7, 2, 2, lca_iters=3, return_all=True)
            inputs = torch.randn(1, 3, 100, 100)
            _, recon, _, _, _ = lca(inputs)
            assert_close(inputs.shape, recon.shape, rtol=0, atol=0)

    def test_LCAConv3D_recon_shape_stride_2_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10, 3, tmp_dir, 5, 7, 9, 2, 2, 2, lca_iters=3, return_all=True
            )
            inputs = torch.randn(1, 3, 8, 100, 100)
            _, recon, _, _, _ = lca(inputs)
            assert_close(inputs.shape, recon.shape, rtol=0, atol=0)

    def test_LCAConv1D_recon_shape_stride_4_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5, 4, lca_iters=3, return_all=True)
            inputs = torch.randn(1, 3, 100)
            _, recon, _, _, _ = lca(inputs)
            assert_close(recon.shape, inputs.shape, rtol=0, atol=0)

    def test_LCAConv2D_recon_shape_stride_4_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 5, 7, 4, 4, lca_iters=3, return_all=True)
            inputs = torch.randn(1, 3, 100, 100)
            _, recon, _, _, _ = lca(inputs)
            assert_close(inputs.shape, recon.shape, rtol=0, atol=0)

    def test_LCAConv3D_recon_shape_stride_4_pad_same(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10, 3, tmp_dir, 5, 7, 9, 4, 4, 4, lca_iters=3, return_all=True
            )
            inputs = torch.randn(1, 3, 8, 100, 100)
            _, recon, _, _, _ = lca(inputs)
            assert_close(inputs.shape, recon.shape, rtol=0, atol=0)

    def test_LCAConv1D_recon_shape_stride_1_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                10, 3, tmp_dir, 100, pad="valid", lca_iters=3, return_all=True
            )
            inputs = torch.randn(1, 3, 100)
            _, recon, _, _, _ = lca(inputs)
            self.assertTupleEqual(recon.numpy().shape, (1, 3, 100))

    def test_LCAConv2D_recon_shape_stride_1_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10, 3, tmp_dir, 100, 100, lca_iters=3, return_all=True, pad="valid"
            )
            inputs = torch.randn(1, 3, 100, 100)
            _, recon, _, _, _ = lca(inputs)
            self.assertTupleEqual(recon.numpy().shape, (1, 3, 100, 100))

    def test_LCAConv3D_recon_shape_stride_1_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10, 3, tmp_dir, 20, 20, 4, pad="valid", lca_iters=3, return_all=True
            )
            inputs = torch.randn(1, 3, 4, 20, 20)
            _, recon, _, _, _ = lca(inputs)
            self.assertTupleEqual(recon.numpy().shape, (1, 3, 4, 20, 20))

    def test_LCAConv1D_code_shape_stride_1_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 100, pad="valid", lca_iters=3)
            inputs = torch.randn(1, 3, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 1))

    def test_LCAConv2D_code_shape_stride_1_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 100, 100, lca_iters=3, pad="valid")
            inputs = torch.randn(1, 3, 100, 100)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 1, 1))

    def test_LCAConv3D_code_shape_stride_1_pad_valid(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 20, 20, 4, pad="valid", lca_iters=3)
            inputs = torch.randn(1, 3, 4, 20, 20)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 1, 1, 1))

    def test_LCAConv3D_code_shape_no_time_pad(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 7, 7, 5, lca_iters=3, no_time_pad=True)
            inputs = torch.randn(1, 3, 5, 20, 20)
            code = lca(inputs)
            self.assertTupleEqual(code.numpy().shape, (1, 10, 1, 20, 20))

    def test_LCAConv3D_recon_shape_no_time_pad(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10, 3, tmp_dir, 7, 7, 5, lca_iters=3, no_time_pad=True, return_all=True
            )
            inputs = torch.randn(1, 3, 5, 20, 20)
            _, recon, _, _, _ = lca(inputs)
            self.assertTupleEqual(recon.numpy().shape, inputs.numpy().shape)

    def test_LCAConv1D_gradient(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 5, lca_iters=3, req_grad=True)
            inputs = torch.randn(1, 3, 100)
            with torch.no_grad():
                code = lca(inputs)

            loss = code.sum()
            with self.assertRaises(RuntimeError):
                loss.backward()

            code = lca(inputs)
            loss = code.sum()
            loss.backward()

    def test_LCAConv2D_gradient(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 5, 7, lca_iters=3, req_grad=True)
            inputs = torch.randn(1, 3, 20, 20)
            with torch.no_grad():
                code = lca(inputs)

            loss = code.sum()
            with self.assertRaises(RuntimeError):
                loss.backward()

            code = lca(inputs)
            loss = code.sum()
            loss.backward()

    def test_LCAConv3D_gradient(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 5, 7, lca_iters=3, req_grad=True)
            inputs = torch.randn(1, 3, 5, 20, 20)
            with torch.no_grad():
                code = lca(inputs)

            loss = code.sum()
            with self.assertRaises(RuntimeError):
                loss.backward()

            code = lca(inputs)
            loss = code.sum()
            loss.backward()

    def test_LCAConv1D_code_feature_as_input(self):
        with TemporaryDirectory() as tmp_dir:
            for lambda_ in torch.arange(0.1, 1.0, 0.1):
                lca = LCAConv1D(
                    10, 3, tmp_dir, 100, pad="valid", input_norm=False, lambda_=lambda_
                )
                inputs = lca.get_weights()[0].unsqueeze(0)
                code = lca(inputs)
                code = code.squeeze()
                code = torch.sort(code, descending=True, stable=True)[0]
                self.assertEqual(torch.count_nonzero(code), 1)
                assert_close(code[0], code.max())

    def test_LCAConv1D_recon_close_to_input_feature_as_input(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                10,
                3,
                tmp_dir,
                100,
                pad="valid",
                input_norm=False,
                lambda_=0.1,
                return_all=True,
            )
            inputs = lca.get_weights()[0].unsqueeze(0)
            _, recon, _, _, _ = lca(inputs)
            mae = (inputs - recon).abs().mean().item()
            self.assertLess(mae, 5.5e-3)

    def test_LCAConv2D_code_feature_as_input(self):
        with TemporaryDirectory() as tmp_dir:
            for lambda_ in torch.arange(0.1, 1.0, 0.1):
                lca = LCAConv2D(
                    10,
                    3,
                    tmp_dir,
                    10,
                    10,
                    pad="valid",
                    input_norm=False,
                    lambda_=lambda_,
                )
                inputs = lca.get_weights()[0].unsqueeze(0)
                code = lca(inputs)
                code = code.squeeze()
                code = torch.sort(code, descending=True, stable=True)[0]
                self.assertEqual(torch.count_nonzero(code), 1)
                assert_close(code[0], code.max())

    def test_LCAConv2D_recon_close_to_input_feature_as_input(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10,
                3,
                tmp_dir,
                10,
                10,
                pad="valid",
                input_norm=False,
                lambda_=0.1,
                return_all=True,
            )
            inputs = lca.get_weights()[0].unsqueeze(0)
            _, recon, _, _, _ = lca(inputs)
            mae = (inputs - recon).abs().mean().item()
            self.assertLess(mae, 5.1e-3)

    def test_LCAConv3D_code_feature_as_input(self):
        with TemporaryDirectory() as tmp_dir:
            for lambda_ in torch.arange(0.1, 1.0, 0.1):
                lca = LCAConv3D(
                    10,
                    3,
                    tmp_dir,
                    10,
                    10,
                    10,
                    pad="valid",
                    input_norm=False,
                    lambda_=lambda_,
                )
                inputs = lca.get_weights()[0].unsqueeze(0)
                code = lca(inputs)
                code = code.squeeze()
                code = torch.sort(code, descending=True, stable=True)[0]
                self.assertTrue(torch.count_nonzero(code), 1)
                assert_close(code[0], code.max())

    def test_LCAConv3D_recon_close_to_input_feature_as_input(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10,
                3,
                tmp_dir,
                10,
                10,
                10,
                pad="valid",
                input_norm=False,
                lambda_=0.1,
                return_all=True,
            )
            inputs = lca.get_weights()[0].unsqueeze(0)
            _, recon, _, _, _ = lca(inputs)
            mae = (inputs - recon).abs().mean().item()
            self.assertLess(mae, 5.5e-3)

    def test_LCAConv1D_compute_lateral_connectivity_stride_1_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 50, 2):
                lca = LCAConv1D(15, 3, tmp_dir, ksize)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, 0].numpy().shape, (15, 15, ksize * 2 - 1)
                )

    def test_LCAConv1D_compute_lateral_connectivity_stride_1_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 52, 2):
                lca = LCAConv1D(15, 3, tmp_dir, ksize, pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, 0].numpy().shape, (15, 15, ksize * 2 - 1)
                )

    def test_LCAConv1D_compute_lateral_connectivity_stride_2_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 50, 2):
                lca = LCAConv1D(15, 3, tmp_dir, ksize, 2)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(conns[..., 0, 0].numpy().shape, (15, 15, ksize))

    def test_LCAConv1D_compute_lateral_connectivity_stride_2_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 52, 2):
                lca = LCAConv1D(15, 3, tmp_dir, ksize, 2, pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(conns[..., 0, 0].numpy().shape, (15, 15, ksize - 1))

    def test_LCAConv1D_compute_lateral_connectivity_odd_ksize_various_strides(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 7
            for stride, exp_size in zip(range(1, 8), [13, 7, 5, 3, 3, 3, 1]):
                lca = LCAConv1D(15, 3, tmp_dir, ksize, stride)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(conns[..., 0, 0].numpy().shape, (15, 15, exp_size))

    def test_LCAConv1D_compute_lateral_connectivity_even_ksize_various_strides(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 8
            for stride, exp_size in zip(range(1, 9), [15, 7, 5, 3, 3, 3, 3, 1]):
                lca = LCAConv1D(15, 3, tmp_dir, ksize, stride, pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(conns[..., 0, 0].numpy().shape, (15, 15, exp_size))

    def test_LCAConv2D_compute_lateral_connectivity_stride_1_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 50, 2):
                ksize2 = ksize + 2
                lca = LCAConv2D(15, 3, tmp_dir, ksize, ksize2)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, :, :].numpy().shape,
                    (15, 15, ksize * 2 - 1, ksize2 * 2 - 1),
                )

    def test_LCAConv2D_compute_lateral_connectivity_stride_1_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 52, 2):
                ksize2 = ksize + 2
                lca = LCAConv2D(15, 3, tmp_dir, ksize, ksize2, pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, :, :].numpy().shape,
                    (15, 15, ksize * 2 - 1, ksize2 * 2 - 1),
                )

    def test_LCAConv2D_compute_lateral_connectivity_stride_2_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 50, 2):
                ksize2 = ksize + 2
                lca = LCAConv2D(15, 3, tmp_dir, ksize, ksize2, 2, 2)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, :, :].numpy().shape, (15, 15, ksize, ksize2)
                )

    def test_LCAConv2D_compute_lateral_connectivity_stride_2_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 52, 2):
                ksize2 = ksize + 2
                lca = LCAConv2D(15, 3, tmp_dir, ksize, ksize2, 2, 2, pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, :, :].numpy().shape, (15, 15, ksize - 1, ksize2 - 1)
                )

    def test_LCAConv2D_compute_lateral_connectivity_odd_ksize_various_strides(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 7
            for stride, exp_size in zip(range(1, 8), [13, 7, 5, 3, 3, 3, 1]):
                lca = LCAConv2D(15, 3, tmp_dir, ksize, ksize, stride, stride)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, :, :].numpy().shape, (15, 15, exp_size, exp_size)
                )

    def test_LCAConv2D_compute_lateral_connectivity_even_ksize_various_strides(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 8
            for stride, exp_size in zip(range(1, 9), [15, 7, 5, 3, 3, 3, 3, 1]):
                lca = LCAConv2D(
                    15, 3, tmp_dir, ksize, ksize, stride, stride, pad="valid"
                )
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns[..., 0, :, :].numpy().shape, (15, 15, exp_size, exp_size)
                )

    def test_LCAConv3D_compute_lateral_connectivity_stride_1_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 11, 2):
                ksize2 = ksize + 2
                ksize3 = ksize + 4
                lca = LCAConv3D(15, 3, tmp_dir, ksize, ksize2, ksize3)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns.numpy().shape,
                    (15, 15, ksize3 * 2 - 1, ksize * 2 - 1, ksize2 * 2 - 1),
                )

    def test_LCAConv3D_compute_lateral_connectivity_stride_1_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 12, 2):
                ksize2 = ksize + 2
                ksize3 = ksize + 4
                lca = LCAConv3D(15, 3, tmp_dir, ksize, ksize2, ksize3, pad="valid")
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns.numpy().shape,
                    (15, 15, ksize3 * 2 - 1, ksize * 2 - 1, ksize2 * 2 - 1),
                )

    def test_LCAConv3D_compute_lateral_connectivity_stride_2_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(1, 11, 2):
                ksize2 = ksize + 2
                ksize3 = ksize + 4
                lca = LCAConv3D(15, 3, tmp_dir, ksize, ksize2, ksize3, 2, 2, 2)
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(conns.numpy().shape, (15, 15, ksize3, ksize, ksize2))

    def test_LCAConv3D_compute_lateral_connectivity_stride_2_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize in range(2, 12, 2):
                ksize2 = ksize + 2
                ksize3 = ksize + 4
                lca = LCAConv3D(
                    15, 3, tmp_dir, ksize, ksize2, ksize3, 2, 2, 2, pad="valid"
                )
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns.numpy().shape, (15, 15, ksize3 - 1, ksize - 1, ksize2 - 1)
                )

    def test_LCAConv3D_compute_lateral_connectivity_odd_ksize_various_strides(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 7
            for stride, exp_size in zip(range(1, 8), [13, 7, 5, 3, 3, 3, 1]):
                lca = LCAConv3D(
                    15, 3, tmp_dir, ksize, ksize, ksize, stride, stride, stride
                )
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns.numpy().shape, (15, 15, exp_size, exp_size, exp_size)
                )

    def test_LCAConv3D_compute_lateral_connectivity_even_ksize_various_strides(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 8
            for stride, exp_size in zip(range(1, 9), [15, 7, 5, 3, 3, 3, 3, 1]):
                lca = LCAConv3D(
                    15,
                    3,
                    tmp_dir,
                    ksize,
                    ksize,
                    ksize,
                    stride,
                    stride,
                    stride,
                    pad="valid",
                )
                conns = lca.compute_lateral_connectivity(lca.weights.detach())
                self.assertEqual(
                    conns.numpy().shape, (15, 15, exp_size, exp_size, exp_size)
                )

    def test_LCAConv3D_no_time_pad(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 7, 7, 7)
            self.assertEqual(lca.input_pad[0], 3)
            lca = LCAConv3D(10, 3, tmp_dir, 7, 7, 7, no_time_pad=True)
            self.assertEqual(lca.input_pad[0], 0)

    def test_l1_norm_of_code_decreases_with_increasing_lambda(self):
        with TemporaryDirectory() as tmp_dir:
            l1_norms = []
            for lambda_ in torch.arange(0.1, 1.0, 0.1):
                lca = LCAConv2D(
                    10,
                    3,
                    tmp_dir,
                    10,
                    10,
                    lambda_=lambda_,
                    pad="valid",
                    input_norm=False,
                )
                inputs = lca.get_weights()[0].unsqueeze(0)
                code = lca(inputs)
                l1_norms.append(code.norm(1).item())
            self.assertEqual(l1_norms, sorted(l1_norms, reverse=True))

    def test_recon_error_increases_with_increasing_lambda(self):
        with TemporaryDirectory() as tmp_dir:
            errors = []
            for lambda_ in torch.arange(0.1, 1.1, 0.1):
                lca = LCAConv2D(
                    10,
                    3,
                    tmp_dir,
                    10,
                    10,
                    lambda_=lambda_,
                    pad="valid",
                    input_norm=False,
                    return_all=True,
                )
                inputs = lca.get_weights()[0].unsqueeze(0)
                _, _, recon_error, _, _ = lca(inputs)
                errors.append(0.5 * recon_error.norm(2) ** 2)
            self.assertEqual(errors, sorted(errors))

    def test_inputs_equal_recon_error_plus_recon_LCAConv1D(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(
                10, 5, tmp_dir, 5, lca_iters=3, input_norm=False, return_all=True
            )
            inputs = torch.randn(3, 5, 100)
            _, recon, recon_error, _, _ = lca(inputs)
            assert_close(inputs, recon_error + recon)

    def test_inputs_equal_recon_error_plus_recon_LCAConv2D(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(
                10, 5, tmp_dir, 5, 5, lca_iters=3, input_norm=False, return_all=True
            )
            inputs = torch.randn(3, 5, 100, 100)
            _, recon, recon_error, _, _ = lca(inputs)
            assert_close(inputs, recon_error + recon)

    def test_inputs_equal_recon_error_plus_recon_LCAConv3D(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(
                10,
                5,
                tmp_dir,
                5,
                5,
                3,
                2,
                2,
                1,
                lca_iters=3,
                input_norm=False,
                return_all=True,
            )
            inputs = torch.randn(3, 5, 10, 100, 100)
            _, recon, recon_error, _, _ = lca(inputs)
            assert_close(inputs, recon_error + recon)

    def test_LCAConv2D_check_conv_params_raises_AssertionError_odd_even_ksizes(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize1 in range(2, 12, 2):
                for ksize2 in range(3, 12, 2):
                    with self.assertRaises(AssertionError):
                        lca = LCAConv2D(10, 1, tmp_dir, ksize1, ksize2)

    def test_LCAConv3D_check_conv_params_raises_AssertionError_odd_even_ksizes(self):
        with TemporaryDirectory() as tmp_dir:
            for ksize1 in range(2, 12, 2):
                for ksize2 in range(3, 12, 2):
                    for ksize3 in range(2, 12, 2):
                        with self.assertRaises(AssertionError):
                            lca = LCAConv3D(10, 1, tmp_dir, ksize1, ksize2, ksize3)

    def test_LCAConv1D_compute_inhib_pad_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 8
            for stride, exp_size in zip(range(1, 9), [7, 6, 6, 4, 5, 6, 7, 0]):
                lca = LCAConv1D(10, 3, tmp_dir, ksize, stride, pad="valid")
                self.assertEqual(lca.lat_conn_pad[0], exp_size)

    def test_LCAConv1D_compute_inhib_pad_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 9
            for stride, exp_size in zip(range(1, 10), [8, 8, 6, 8, 5, 6, 7, 8, 0]):
                lca = LCAConv1D(10, 3, tmp_dir, ksize, stride, pad="valid")
                self.assertEqual(lca.lat_conn_pad[0], exp_size)

    def test_LCAConv2D_compute_inhib_pad_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 8
            for stride, exp_size in zip(range(1, 9), [7, 6, 6, 4, 5, 6, 7, 0]):
                lca = LCAConv2D(
                    10, 3, tmp_dir, ksize, ksize, stride, stride, pad="valid"
                )
                self.assertEqual(lca.lat_conn_pad[1:], (exp_size, exp_size))

    def test_LCAConv2D_compute_inhib_pad_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 9
            for stride, exp_size in zip(range(1, 10), [8, 8, 6, 8, 5, 6, 7, 8, 0]):
                lca = LCAConv2D(
                    10, 3, tmp_dir, ksize, ksize, stride, stride, pad="valid"
                )
                self.assertEqual(lca.lat_conn_pad[1:], (exp_size, exp_size))

    def test_LCAConv3D_compute_inhib_pad_even_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 8
            for stride, exp_size in zip(range(1, 9), [7, 6, 6, 4, 5, 6, 7, 0]):
                lca = LCAConv3D(
                    10,
                    3,
                    tmp_dir,
                    ksize,
                    ksize,
                    ksize,
                    stride,
                    stride,
                    stride,
                    pad="valid",
                )
                self.assertEqual(lca.lat_conn_pad, (exp_size,) * 3)

    def test_LCAConv3D_compute_inhib_pad_odd_ksize(self):
        with TemporaryDirectory() as tmp_dir:
            ksize = 9
            for stride, exp_size in zip(range(1, 10), [8, 8, 6, 8, 5, 6, 7, 8, 0]):
                lca = LCAConv3D(
                    10,
                    3,
                    tmp_dir,
                    ksize,
                    ksize,
                    ksize,
                    stride,
                    stride,
                    stride,
                    pad="valid",
                )
                self.assertEqual(lca.lat_conn_pad, (exp_size,) * 3)

    def test_LCAConv1D_compute_inhib_pad_ksize_equal_1(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 1)
            self.assertEqual(lca.lat_conn_pad, (0, 0, 0))

    def test_LCAConv2D_compute_inhib_pad_ksize_equal_1(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 1, 1)
            self.assertEqual(lca.lat_conn_pad, (0, 0, 0))

    def test_LCAConv3D_compute_inhib_pad_ksize_equal_1(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 1, 1, 1)
            self.assertEqual(lca.lat_conn_pad, (0, 0, 0))

    def test_LCAConv1D_compute_lateral_connectivity_ksize_equal_1(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv1D(10, 3, tmp_dir, 1)
            conns = lca.compute_lateral_connectivity(lca.weights)
            self.assertEqual(conns.numpy().shape, (10, 10, 1, 1, 1))

    def test_LCAConv2D_compute_lateral_connectivity_ksize_equal_1(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv2D(10, 3, tmp_dir, 1, 1)
            conns = lca.compute_lateral_connectivity(lca.weights)
            self.assertEqual(conns.numpy().shape, (10, 10, 1, 1, 1))

    def test_LCAConv3D_compute_lateral_connectivity_ksize_equal_1(self):
        with TemporaryDirectory() as tmp_dir:
            lca = LCAConv3D(10, 3, tmp_dir, 1, 1, 1)
            conns = lca.compute_lateral_connectivity(lca.weights)
            self.assertEqual(conns.numpy().shape, (10, 10, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
