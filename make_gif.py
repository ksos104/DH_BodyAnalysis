import os
from PIL import Image
import cv2


def main():
    root = '/mnt/server8_hard3/msson/Self-Correction-Human-Parsing/result_for_gifs'
    dir_list = ['vid0_real', 'vid0_gt', 'vid0_pred',
                'vid1_real', 'vid1_gt', 'vid1_pred']

    for dir_name in dir_list:
        dir_path = os.path.join(root, dir_name)
        frame_name_list = os.listdir(dir_path)
        frame_name_list.sort()
        frame_list = []
        for frame_name in frame_name_list:
            frame_path = os.path.join(dir_path, frame_name)
            frame_list.append(Image.fromarray(cv2.imread(frame_path, cv2.IMREAD_COLOR)[..., ::-1]))

        frame_one = frame_list[0]
        save_name = dir_name + '.gif'
        save_path = os.path.join(root, save_name)
        frame_one.save(save_path, format='GIF', append_images=frame_list, save_all=True, duration=100, loop=0)
    

if __name__ == "__main__":
    main()