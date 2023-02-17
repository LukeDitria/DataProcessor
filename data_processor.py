import argparse
import json
import os
from multiprocessing import Pool
import cv2
import numpy as np
import time


class Extractor:
    def __init__(self, root_source_dir, root_save_dir, valid_file_ext=None, file_out_ext="json",
                 restart_job=False, max_processes=None):
        if valid_file_ext is None:
            valid_file_ext = ["csv"]
        self.root_source_dir = root_source_dir
        self.root_save_dir = root_save_dir
        self.restart_job = restart_job
        self.valid_file_ext = valid_file_ext
        self.file_out_ext = file_out_ext
        self.save_dir = os.path.join(self.root_save_dir, self.root_source_dir.split("/")[-1])
        self.max_processes = max_processes

    def process_files(self):
        total_files = sum([len(files) for root, dirs, files in os.walk(self.root_source_dir)])
        print("Total number of files found: %d" % total_files)
        print('Processing files with %d processes!' % self.max_processes)
        files_processed = 0

        # Use os.walk() to loop through the nested directories
        with Pool(processes=self.max_processes) as pool:
            for root, dirs, files in os.walk(self.root_source_dir):

                if len(files) > 0:
                    sub_dir = self.save_dir + root.replace(self.root_source_dir, '')
                    # Loop through the list of files
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_type = file.split(".")[-1]

                        # Check if file is a valid file
                        if self.filter_filename(file):
                            continue

                        new_file_name = "_".join(file.split(".")[:-1]) + self.file_out_ext
                        new_file_path = os.path.join(sub_dir, new_file_name)

                        # Check if file already exists or job needs to be restarted
                        if not os.path.isfile(new_file_path) or self.restart_job:
                            if file_type.lower() in self.valid_file_ext:
                                if not os.path.isdir(sub_dir):
                                    os.makedirs(sub_dir)

                                # Use multiprocessing to extract data from multiple files at once
                                pool.apply_async(self.extract_data, args=(file_path, new_file_path))

                        files_processed += 1
                        completion_percentage = (files_processed / total_files) * 100
                        print(f"Completion percentage: {completion_percentage:.2f}%")

        print('COMPLETE!')
        print('SAVED TO %s' % self.root_save_dir)

    def filter_filename(self, file):
        return False

    def extract_data(self, file_path, new_file_path):
        # PROCESS DATA #
        img = cv2.imread(file_path)
        data_out = {"img_mean": np.mean(img),
                    "img_var": np.var(img)}

        self.save_data(data_out, new_file_path)

    def save_data(self, data_out, save_file_path):
        # Save the data dict as a json file -> should make this customisable
        with open(save_file_path, 'w') as fp:
            json.dump(data_out, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_source_dir', '-srd', type=str, default='.', help='root source dir')
    parser.add_argument('--root_save_dir', '-svd', type=str, default='.', help='root save dir')
    parser.add_argument('--valid_file_ext', '-vfe', nargs='+', default=['csv'], help='valid file extensions')
    parser.add_argument('--file_out_ext', '-foe', type=str, default='json', help='file output extension')
    parser.add_argument('--restart_job', '-rj', action='store_true', help='overwrite all previous extractions')
    parser.add_argument('--max_processes', '-mp', type=int, default=4, help='maximum number of processes')
    args = parser.parse_args()

    # Create an instance of Extractor class
    extractor = Extractor(args.root_source_dir, args.root_save_dir, args.valid_file_ext, args.file_out_ext,
                          args.restart_job, args.max_processes)
    start_time = time.time()
    extractor.process_files()
    print("Time to process %.2f secs" % (time.time() - start_time))
