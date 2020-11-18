import os

from lib.processing_utils import frame_to_face

frame_dir = '/home/shicaiwei/data_set_make/blue'
face_dir = '/home/shicaiwei/data_set_make/blue_face'
frame_to_face(frame_dir=frame_dir, face_dir=face_dir, model_name='TF', save_mode='.jpg')
