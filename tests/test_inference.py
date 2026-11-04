import numpy as np
import torch

from tennis_contact.inference import restore_logits


def test_restore_averages_duplicate_native_frames_before_interpolation():
    # 重复采样若直接插值，会选中最后一次输出而非平均。
    windows = [{"indices": np.array([0, 0, 2]), "core_mask": np.ones(3, dtype=bool)}]
    outputs = [{"eventness_logit": torch.tensor([1., 3., 8.]),
                "type_logits": torch.tensor([[2., 0.], [4., 2.], [9., 7.]])}]
    result = restore_logits(windows, outputs, np.array([0., .02, .06]))
    np.testing.assert_allclose(result[:, 0], [2., 4., 8.])
    np.testing.assert_allclose(result[:, 1:], [[3., 1.], [5., 3.], [9., 7.]])


def test_restore_excludes_halo_outputs_and_extends_endpoint_values():
    windows = [{"indices": np.array([0, 1, 2]), "core_mask": np.array([False, True, False])}]
    outputs = [{"eventness_logit": torch.tensor([99., 2., 99.]),
                "type_logits": torch.tensor([[99., 99.], [3., 4.], [99., 99.]])}]
    result = restore_logits(windows, outputs, np.array([0., .03, .06]))
    np.testing.assert_array_equal(result, [[2., 3., 4.]] * 3)
