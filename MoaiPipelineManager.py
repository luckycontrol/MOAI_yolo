import yaml

class Manager:
    def __init__(self, project, subproject, task, version, location="d:/moai_test"):
        self.project = project
        self.subproject = subproject
        self.task = task
        self.version = version
        self.location = location

    def get_data_yaml(self):
        data_yaml_path = f"{self.location}/{self.project}/{self.subproject}/{self.task}/dataset/train_dataset/data.yaml"
        with open(data_yaml_path, "r") as f:
            data = yaml.safe_load(f)

        return data

    def get_data_yaml_path(self):
        data_yaml_path = f"{self.location}/{self.project}/{self.subproject}/{self.task}/dataset/train_dataset/data.yaml"
        return data_yaml_path

    def get_hyp_yaml(self):
        hyp_yaml_path = f"{self.location}/{self.project}/{self.subproject}/{self.task}/dataset/train_dataset/hyp.yaml"
        with open(hyp_yaml_path, "r") as f:
            hyp = yaml.safe_load(f)

        return hyp

    def get_hyp_yaml_path(self):
        hyp_yaml_path = f"{self.location}/{self.project}/{self.subproject}/{self.task}/dataset/train_dataset/hyp.yaml"
        return hyp_yaml_path

    def get_weight_folder_path(self):
        weight_folder_path = f"{self.location}/{self.project}/{self.subproject}/{self.task}/{self.version}/weights"
        return weight_folder_path

    def get_best_weight_path(self):
        best_weight_path = f"{self.location}/{self.project}/{self.subproject}/{self.task}/{self.version}/weights/best.pt"
        return best_weight_path

    def get_train_dataset_path(self):
        train_dataset_path = f"{self.location}/{self.project}/{self.subproject}/{self.task}/dataset/train_dataset"
        return train_dataset_path

    def get_test_dataset_path(self):
        test_dataset_path = f"{self.location}/{self.project}/{self.subproject}/{self.task}/{self.version}/inference_dataset"
        return test_dataset_path

    def get_version_folder_path(self):
        version_folder_path = f"{self.location}/{self.project}/{self.subproject}/{self.task}/{self.version}"
        return version_folder_path

    def get_training_result_folder_path(self):
        training_result_folder_path = f"{self.location}/{self.project}/{self.subproject}/{self.task}/{self.version}/training_result"
        return training_result_folder_path

    def get_test_result_folder_path(self):
        test_result_folder_path = f"{self.location}/{self.project}/{self.subproject}/{self.task}/{self.version}/inference_result"
        return test_result_folder_path