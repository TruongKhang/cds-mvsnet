import torch


class PriorState(object):
    def __init__(self, max_size=4):
        self.depths = None
        self.confs = None
        # self.proj_matrices = None
        self.max_size = max_size
        self.length = 0

    def size(self):
        return self.depths["stage1"].size(1) if self.depths is not None else 0

    def update(self, depth, conf):
        if self.depths is None:
            self.depths = {stage: depth[stage].unsqueeze(1) for stage in depth.keys()}
            self.confs = {stage: conf[stage].unsqueeze(1) for stage in conf.keys()}
            # self.proj_matrices = {stage: proj_matrix[stage].unsqueeze(1) for stage in proj_matrix.keys()}
        else:
            for stage in self.depths.keys():
                self.depths[stage] = torch.cat((depth[stage].unsqueeze(1), self.depths[stage][:, :(self.max_size-1), ...]), dim=1)
                self.confs[stage] = torch.cat((conf[stage].unsqueeze(1), self.confs[stage][:, :(self.max_size-1), ...]), dim=1)
                # self.proj_matrices[stage] = torch.cat((proj_matrix[stage].unsqueeze(1), self.proj_matrices[stage][:, :self.max_size, ...]), dim=1)

    def get(self):
        return self.depths, self.confs

    def reset(self):
        self.depths, self.confs = None, None
