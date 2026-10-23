import base64
import zlib

import cv2
import numpy as np

from cone_validation.distance import estimate_distance
from cone_validation.fsoco import Cone, decode_bitmap
from cone_validation.keypoints import keypoint_errors, match_to_masks, order_ok, outline_distance

SQUARE = [np.array([[0, 0], [10, 0], [10, 20], [0, 20]], float)]


def test_outline_distance_inside_and_outside():
    assert np.isclose(outline_distance((0, 10), SQUARE), 0)
    assert np.isclose(outline_distance((5, 10), SQUARE), 5)      # inside, 5 px from the nearest edge
    assert np.isclose(outline_distance((13, 10), SQUARE), 3)     # outside


def test_errors_relative_to_cone_height():
    cone = Cone(cls=2, box=np.array([0, 0, 10, 20], float), contours=SQUARE)
    px, rel = keypoint_errors(np.array([[0, 10], [5, 10]]), cone)
    assert np.allclose(px, [0, 5]) and np.allclose(rel, [0, 0.25])


def test_order_check_catches_mirrored_keypoints():
    good = np.array([[0, 20], [1, 15], [2, 5], [3, 0], [7, 0], [8, 5], [9, 15], [10, 20]], float)
    assert order_ok(good)
    assert not order_ok(good[::-1])


def test_mask_matching_is_one_to_one_and_same_class():
    cones = [Cone(2, np.array([0, 0, 10, 20], float)), Cone(3, np.array([0, 0, 10, 20], float))]
    pairs = match_to_masks(np.array([2, 2]), np.array([[0, 0, 10, 20], [0, 0, 10, 20]], float), [0.9, 0.8], cones)
    assert pairs == [(0, 0)]


def test_decode_supervisely_bitmap():
    mask = np.zeros((4, 3), np.uint8)
    mask[1:3, 1] = 1
    rgba = np.zeros((4, 3, 4), np.uint8)
    rgba[..., 3] = mask * 255
    data = base64.b64encode(zlib.compress(cv2.imencode(".png", rgba)[1].tobytes())).decode()
    assert (decode_bitmap(data) == mask.astype(bool)).all()


def test_distance_from_height():
    # 0.325 m cone, 100 px tall, 1000 px image with a 90 degree vertical FOV: f = 500 px
    assert np.isclose(estimate_distance(100, 0.325, 1000, 90), 1.625)


def test_outline_lies_on_pixel_edges_on_every_side():
    from cone_validation.fsoco import outline
    mask = np.zeros((20, 10), bool)
    mask[:, :] = True                                   # pixels 0..9 wide, 0..19 tall: edges at x=0,10 and y=0,20
    contours = outline(mask, (100, 50))
    for point in [(100, 60), (110, 60), (105, 50), (105, 70)]:
        assert outline_distance(point, contours) <= 0.13


def test_order_check_catches_a_swap_on_one_side():
    good = np.array([[0, 20], [1, 15], [2, 5], [3, 0], [7, 0], [8, 5], [9, 15], [10, 20]], float)
    assert not order_ok(good[[0, 2, 1, 3, 4, 5, 6, 7]])
