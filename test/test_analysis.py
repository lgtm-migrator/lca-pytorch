import unittest

import torch

from lcapt.analysis import make_feature_grid


class TestAnalysis(unittest.TestCase):
    def test_make_feature_grid_returns_correct_shape_3D_input_one_channel(self):
        inputs = torch.randn(5, 1, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (5, 10))

    def test_make_feature_grid_returns_correct_shape_3D_input_with_multi_channel(self):
        inputs = torch.randn(5, 3, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (17, 26))

    def test_make_feature_grid_returns_correct_shape_4D_input(self):
        inputs = torch.randn(5, 3, 10, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (38, 26, 3))

    def test_make_feature_grid_returns_correct_shape_5D_input_time_equals_one(self):
        inputs = torch.randn(5, 3, 1, 10, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (38, 26, 3))

    def test_make_feature_grid_returns_correct_shape_5D_input_time_gt_one(self):
        inputs = torch.randn(5, 3, 3, 10, 10)
        grid = make_feature_grid(inputs)
        self.assertEqual(grid.numpy().shape, (3, 38, 26, 3))


if __name__ == "__main__":
    unittest.main()
