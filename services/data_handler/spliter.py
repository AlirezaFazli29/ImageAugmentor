import os
import shutil
from tqdm.notebook import tqdm
from typing import Optional, List
from sklearn.model_selection import train_test_split


class DatasetSplitter():

    def __init__(
            self,
            train_size: Optional[float] = None,
            valid_size: Optional[float] = 0.2,
            test_size: Optional[float] = 0.1,
    ):
        
        if train_size==None:
            train_size = 1 - (valid_size + test_size)

        sizes = {
            'train_size': train_size,
            'valid_size': valid_size,
            'test_size': test_size,
        }
        
        for name, value in sizes.items():
            if value is not None and not (0 < value < 1):
                raise ValueError(
                    f"{name} must be between 0 and 1 (exclusive), got {value}"
                )
            
        total = sum(v for v in sizes.values() if v is not None)
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Sum of split sizes must be exactly 1.0, got {total}")
        
        self.train_size = train_size
        self.temp_size = test_size + valid_size
        self.valid_size = valid_size / self.temp_size
        self.test_size = test_size / self.temp_size
        

    def splitter(
            self,
            names: List,
            random_state: Optional[int] = None,
            shuffle: bool = True,
    ):
        
        train_names, temp_names = train_test_split(
            names,
            test_size=self.temp_size,
            # train_size=self.train_size,
            random_state=random_state,
            shuffle=shuffle,
        )

        valid_names, test_names = train_test_split(
            temp_names,
            test_size=self.test_size,
            # train_size=self.valid_size,
            random_state=random_state,
            shuffle=shuffle,
        )

        return train_names, valid_names, test_names
    

    def pair_splitter(
            self,
            data_names: List,
            label_names: List,
            random_state: Optional[int] = None,
            shuffle: bool = True,
    ):
        
        data_train_names, data_temp_names, labels_train_names, labels_temp_names = train_test_split(
            data_names,
            label_names,
            test_size=self.temp_size,
            # train_size=self.train_size,
            random_state=random_state,
            shuffle=shuffle,
        )

        data_valid_names, data_test_names, labels_valid_names, labels_test_names = train_test_split(
            data_temp_names,
            labels_temp_names,
            test_size=self.test_size,
            # train_size=self.valid_size,
            random_state=random_state,
            shuffle=shuffle,
        )

        return (
            data_train_names, labels_train_names,
            data_valid_names, labels_valid_names,
            data_test_names, labels_test_names,
        )


    def split_pair_from_dir(
            self,
            data_dir,
            label_dir,
            output_dir,
            random_state: Optional[int] = None,
            shuffle: bool = True,
    ):
        
        os.makedirs(output_dir, exist_ok=True)
        data_names = sorted(os.listdir(data_dir))
        label_names = sorted(os.listdir(label_dir))

        if len(data_names) != len(label_names):
            raise ValueError("Mismatched number of data and label files.")

        splits = self.pair_splitter(data_names, label_names, random_state=random_state, shuffle=shuffle)
        data_train, labels_train, data_valid, labels_valid, data_test, labels_test = splits

        sets = ["train", "valid", "test"]
        grouped_data = [
            (data_train, labels_train),
            (data_valid, labels_valid),
            (data_test,  labels_test),
        ]

        split_data = list(zip(sets, grouped_data))

        progress_bar = tqdm(split_data, total=len(split_data), desc="Splitting sets", leave=True)
        for set_name, (data_files, label_files) in progress_bar:
            images_out_dir = os.path.join(output_dir, set_name, "images")
            labels_out_dir = os.path.join(output_dir, set_name, "labels")
            os.makedirs(images_out_dir, exist_ok=True)
            os.makedirs(labels_out_dir, exist_ok=True)

            progress_bar_2 = tqdm(
                list(zip(data_files, label_files)),
                total=len(data_files),
                desc=f"Copying {set_name}",
                leave=False
            )
            for dname, lname in progress_bar_2:
                shutil.copy(os.path.join(data_dir, dname), os.path.join(images_out_dir, dname))
                shutil.copy(os.path.join(label_dir, lname), os.path.join(labels_out_dir, lname))


