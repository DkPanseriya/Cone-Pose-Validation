import numpy as np

from cone_validation.metrics import average_precision, confusion_matrix, match


def boxes(*b):
    return np.array(b, float).reshape(-1, 4)


NO = np.zeros(0, bool)


def test_greedy_match_retries_second_best():
    gt = boxes([0, 0, 10, 10], [6, 0, 16, 10])
    tp, ign, _ = match(boxes([0, 0, 10, 10], [3, 0, 13, 10]), [0.9, 0.8], gt, [False, False], [False, False], [0.4])
    assert tp[:, 0].tolist() == [True, True]


def test_duplicate_is_false_positive():
    tp, ign, _ = match(boxes([0, 0, 10, 10], [0, 0, 10, 10]), [0.9, 0.8], boxes([0, 0, 10, 10]), [False],
                       [False, False], [0.5])
    assert tp[:, 0].tolist() == [True, False] and not ign.any()


def test_prediction_on_ignored_cone_is_neither_tp_nor_fp():
    tp, ign, _ = match(boxes([0, 0, 10, 10]), [0.9], boxes([0, 0, 10, 10]), [True], [False], [0.5])
    assert not tp.any() and ign.all()


def test_unmatched_prediction_outside_gate_is_ignored():
    tp, ign, _ = match(boxes([50, 50, 52, 52]), [0.9], boxes([0, 0, 10, 10]), [False], [True], [0.5])
    assert not tp.any() and ign.all()


def test_ap_skips_ignored_and_handles_missing_class():
    tp = np.array([[True], [False]])
    ign = np.array([[False], [True]])
    assert np.isclose(average_precision(tp, ign, [0.9, 0.8], n_gt=1)[0], 1.0)
    assert np.isnan(average_precision(tp, ign, [0.9, 0.8], n_gt=0)).all()


def test_confusion_matrix_background_and_wrong_colour():
    gt_cls = np.array([2, 3, 0])
    gt = boxes([0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10])
    pred = boxes([0, 0, 10, 10], [20, 0, 30, 10], [100, 100, 110, 110])
    cm, detected = confusion_matrix(np.array([2, 2, 1]), pred, [0.9, 0.8, 0.7], gt_cls, gt,
                                    np.array([False, False, False]), np.array([False, False, False]), n_classes=4)
    assert cm[2, 2] == 1 and cm[2, 3] == 1      # right colour, and a yellow cone called blue
    assert cm[4, 0] == 1 and cm[1, 4] == 1      # a missed cone, and a false positive
    assert detected.tolist() == [2, 2, -1]


def test_confusion_matrix_ignores_duplicates_on_ignored_cones():
    gt = boxes([0, 0, 10, 10])
    pred = boxes([0, 0, 10, 10], [0, 0, 10, 10])
    cm, _ = confusion_matrix(np.array([2, 2]), pred, [0.9, 0.8], np.array([2]), gt, np.array([True]),
                             np.array([False, False]), n_classes=4)
    assert cm.sum() == 0


def test_confusion_matrix_prefers_a_valid_cone_over_an_ignored_neighbour():
    gt = boxes([0, 0, 10, 10], [1, 0, 11, 10])           # second cone ignored, overlaps the first
    cm, detected = confusion_matrix(np.array([2]), boxes([1, 0, 11, 10]), [0.9], np.array([2, 2]), gt,
                                    np.array([False, True]), np.array([False]), n_classes=4)
    assert cm[2, 2] == 1 and detected.tolist() == [2, -1]
