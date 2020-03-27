# Python
import numpy as np
import time

# Custom
import kitti
import batch_loader

# Data Loading Module
import torch.multiprocessing
from torch.multiprocessing import Process, Queue, Value, cpu_count

class BatchSchedulerMP:
    def __init__(self, inputs, mode):
        self.mode = mode
        self.inputs = inputs
        self.queue = Queue()
        self.control = Value('i', 1)
        if self.mode == 0:
            self.process = Process(target=self.worker, args=(self.inputs, self.queue, self.control))
            self.process.start()
        # self.worker(self.inputs, self.queue, self.control)

    def stop(self):
        self.control.value = 0

    def get_mp(self):
        if self.mode == 0:
            while 1:
                items = self.queue.get()
                if items is None: break
                yield items
        else:
            for items in self.single(self.inputs):
                if items is None: break
                yield items

    def load(self, inputs):
        dataset_path = inputs["dataset_path"]
        t_win_r = inputs["t_win_r"]
        img_size = inputs["img_size"]
        crop_w = inputs["crop_w"]
        d_candi = inputs["d_candi"]
        d_candi_up = inputs["d_candi_up"]
        dataset_name = inputs["dataset_name"]
        hack_num = inputs["hack_num"]
        batch_size = inputs["batch_size"]
        qmax = inputs["qmax"]
        n_epoch = inputs["n_epoch"]
        mode = inputs["mode"]
        pytorch_scaling = inputs["pytorch_scaling"]
        velodyne_depth = inputs["velodyne_depth"]
        if mode == "train":
            split_txt = './kitti_split/training.txt'
        elif mode == "val":
            split_txt = './kitti_split/testing.txt'

        dataset_init = kitti.KITTI_dataset
        fun_get_paths = lambda traj_indx: kitti.get_paths(traj_indx, split_txt=split_txt,
                                                          mode=mode,
                                                          database_path_base=dataset_path, t_win=t_win_r)

        # Load Dataset
        n_scenes, _, _, _, _ = fun_get_paths(0)
        traj_Indx = np.arange(0, n_scenes)
        fldr_path, img_paths, dmap_paths, poses, intrin_path = fun_get_paths(0)
        dataset = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path=intrin_path, img_size=img_size, digitize=True,
                               d_candi=d_candi, d_candi_up=d_candi_up, resize_dmap=.25,
                               crop_w=crop_w, velodyne_depth=velodyne_depth, pytorch_scaling=pytorch_scaling)
        BatchScheduler = batch_loader.Batch_Loader(
            batch_size=batch_size, fun_get_paths=fun_get_paths,
            dataset_traj=dataset, nTraj=len(traj_Indx), dataset_name=dataset_name, t_win_r=t_win_r,
            hack_num=hack_num)
        return BatchScheduler

    def worker(self, inputs, queue, control):
        qmax = inputs["qmax"]
        n_epoch = inputs["n_epoch"]

        # Iterate batch
        broken = False
        for iepoch in range(n_epoch):
            BatchScheduler = self.load(inputs)
            for batch_idx in range(len(BatchScheduler)):
                start = time.time()
                for frame_count, ref_indx in enumerate(range(BatchScheduler.traj_len)):
                    local_info = BatchScheduler.local_info_full()

                    # Queue Max
                    while queue.qsize() >= qmax:
                        if control.value == 0:
                            broken = True
                            break
                        time.sleep(0.01)
                    if control.value == 0:
                        broken = True
                        break

                    # Put in Q
                    queue.put([local_info, len(BatchScheduler), batch_idx, frame_count, ref_indx, iepoch])

                    # Update dat_array
                    if frame_count < BatchScheduler.traj_len - 1:
                        BatchScheduler.proceed_frame()

                    # print(batch_idx, frame_count)
                    if broken: break

                if broken: break
                BatchScheduler.proceed_batch()

            if broken: break
        queue.put(None)
        time.sleep(1)

    def single(self, inputs):
        qmax = inputs["qmax"]
        n_epoch = inputs["n_epoch"]

        # Iterate batch
        broken = False
        for iepoch in range(n_epoch):
            BatchScheduler = self.load(inputs)
            for batch_idx in range(len(BatchScheduler)):
                start = time.time()
                for frame_count, ref_indx in enumerate(range(BatchScheduler.traj_len)):
                    local_info = BatchScheduler.local_info_full()

                    # if frame_count == 0:
                    #   print(local_info["src_dats"][0][0]["left_camera"]["img_path"])

                    # Put in Q
                    yield [local_info, len(BatchScheduler), batch_idx, frame_count, ref_indx, iepoch]

                    # Update dat_array
                    if frame_count < BatchScheduler.traj_len - 1:
                        BatchScheduler.proceed_frame()

                    # print(batch_idx, frame_count)
                    if broken: break

                if broken: break
                BatchScheduler.proceed_batch()

            if broken: break
        yield None