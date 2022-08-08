import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChLocalLoss(nn.Module):
    def __init__(self,
                 device,
                 img_rep_size,
                 config
                 ):
        super(ChLocalLoss, self).__init__()
        self.device = device

        self.no_of_decoder_blocks = config.no_of_decoder_blocks
        self.no_of_local_regions = config.no_of_local_regions
        self.no_of_neg_local_regions = config.no_of_neg_local_regions
        self.local_reg_size = config.local_reg_size
        self.img_size_x, self.img_size_y = img_rep_size, img_rep_size
        self.wgt_en = config.wgt_en

        self.bs = 2 * config.batch_size

        self.pos_sample_indexes, self.neg_sample_indexes = self._define_local_regions()

        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.temp_fac = config.temp_fac

    def _define_local_regions(self):
        # define local loss term

        # dimension of feature map in x and y directions (im_x,im_y) defined based on the no. of decoder blocks used.
        # if local_reg_size=1 then local region size is 3x3, local_reg_size=0 then local region size is 1x1.
        if self.no_of_decoder_blocks == 1:
            if self.local_reg_size == 1:
                im_x, im_y = int(self.img_size_x / 16) - 4, int(self.img_size_y / 16) - 4
            else:
                im_x, im_y = int(self.img_size_x / 16) - 1, int(self.img_size_y / 16) - 1
        elif self.no_of_decoder_blocks == 2:
            if self.local_reg_size == 1:
                im_x, im_y = int(self.img_size_x / 8) - 4, int(self.img_size_y / 8) - 4
            else:
                im_x, im_y = int(self.img_size_x / 8) - 1, int(self.img_size_y / 8) - 1
        elif self.no_of_decoder_blocks == 3:
            if self.local_reg_size == 1:
                im_x, im_y = int(self.img_size_x / 4) - 4, int(self.img_size_y / 4) - 4
            else:
                im_x, im_y = int(self.img_size_x / 4) - 1, int(self.img_size_y / 4) - 1
        elif self.no_of_decoder_blocks == 4:
            if self.local_reg_size == 1:
                im_x, im_y = int(self.img_size_x / 2) - 4, int(self.img_size_y / 2) - 4
            else:
                im_x, im_y = int(self.img_size_x / 2) - 1, int(self.img_size_y / 2) - 1
        else:
            if self.local_reg_size == 1:
                im_x, im_y = int(self.img_size_x) - 4, int(self.img_size_y) - 4
            else:
                im_x, im_y = int(self.img_size_x) - 1, int(self.img_size_y) - 1

        if self.no_of_local_regions == 9:
            # Indexes for the local regions to be selected for computing local contrastive loss All the local regions
            # for positive samples from images (x_a1_i,x_a2_i), where x_a1_i,x_a2_i are two augmented versions of x_i.
            pos_sample_indexes = np.zeros((self.no_of_local_regions, 2), dtype=np.int32)
            pos_sample_indexes[0], pos_sample_indexes[1], pos_sample_indexes[2] = [0, 0], [0, int(im_y / 2)], [0, im_y]
            pos_sample_indexes[3], pos_sample_indexes[4], pos_sample_indexes[5] = [int(im_x / 2), 0], [int(im_x / 2),
                                                                                                       int(im_y / 2)], [
                                                                                      int(im_x / 2), im_y]
            pos_sample_indexes[6], pos_sample_indexes[7], pos_sample_indexes[8] = [im_x, 0], [im_x, int(im_y / 2)], [
                im_x,
                im_y]

            # Indexes for negative samples w,r.t a positive sample.
            neg_sample_indexes = np.zeros((self.no_of_local_regions, self.no_of_neg_local_regions, 2), dtype=np.int32)
            # Each positive local region will have corresponding regions that act as negative samples to be
            # contrasted. For each positive sample, we pick the nearby no_of_neg_local_regions (5) local regions as
            # negative samples from both the images (x_a1_i, x_a2_i) for local region at (0,0), define the negative
            # samples co-ordinates accordingly
            neg_sample_indexes[0, :, :] = [[0, int(im_y / 2)], [int(im_x / 2), 0], [int(im_x / 2), int(im_y / 2)],
                                           [0, im_y], [im_x, 0]]
            # similarly, define negative samples co-ordinates according to positive sample
            neg_sample_indexes[1, :, :] = [[0, 0], [0, im_y], [int(im_x / 2), 0], [int(im_x / 2), int(im_y / 2)],
                                           [int(im_x / 2), im_y]]
            neg_sample_indexes[2, :, :] = [[0, 0], [0, int(im_y / 2)], [int(im_x / 2), int(im_y / 2)],
                                           [int(im_x / 2), im_y], [im_x, im_y]]
            neg_sample_indexes[3, :, :] = [[0, 0], [0, int(im_y / 2)], [int(im_x / 2), int(im_y / 2)], [im_x, 0],
                                           [im_x, int(im_y / 2)]]
            neg_sample_indexes[4, :, :] = [[0, 0], [0, int(im_y / 2)], [int(im_x / 2), 0], [int(im_x / 2), im_y],
                                           [im_x, int(im_y / 2)]]
            neg_sample_indexes[5, :, :] = [[0, int(im_y / 2)], [0, im_y], [int(im_x / 2), int(im_y / 2)],
                                           [im_x, int(im_y / 2)], [im_x, im_y]]
            neg_sample_indexes[6, :, :] = [[0, 0], [int(im_x / 2), 0], [int(im_x / 2), int(im_y / 2)],
                                           [im_x, int(im_y / 2)], [im_x, im_y]]
            neg_sample_indexes[7, :, :] = [[int(im_x / 2), 0], [int(im_x / 2), int(im_y / 2)], [int(im_x / 2), im_y],
                                           [im_x, 0], [im_x, im_y]]
            neg_sample_indexes[8, :, :] = [[0, im_y], [int(im_x / 2), int(im_y / 2)], [int(im_x / 2), im_y], [im_x, 0],
                                           [im_x, int(im_y / 2)]]

        elif self.no_of_local_regions == 13 and self.no_of_neg_local_regions == 5:
            # Indexes for the local regions to be selected for computing local contrastive loss All the local regions
            # for positive samples from images (x_a1_i,x_a2_i), where x_a1_i,x_a2_i are two augmented versions of x_i.
            pos_sample_indexes = np.zeros((self.no_of_local_regions, 2), dtype=np.int32)
            pos_sample_indexes[0], pos_sample_indexes[1], pos_sample_indexes[2] = [0, 0], [0, int(im_y / 2)], [0, im_y]
            pos_sample_indexes[3], pos_sample_indexes[4], pos_sample_indexes[5] = [int(im_x / 2), 0], [int(im_x / 2),
                                                                                                       int(im_y / 2)], [
                                                                                      int(im_x / 2), im_y]
            pos_sample_indexes[6], pos_sample_indexes[7], pos_sample_indexes[8] = [im_x, 0], [im_x, int(im_y / 2)], [
                im_x,
                im_y]
            pos_sample_indexes[9], pos_sample_indexes[10] = [int(im_x / 4), int(im_y / 4)], [int(im_x / 4),
                                                                                             int(3 * im_y / 4)]
            pos_sample_indexes[11], pos_sample_indexes[12] = [int(3 * im_x / 4), int(im_y / 4)], [int(3 * im_x / 4),
                                                                                                  int(3 * im_y / 4)]

            # Indexes for negative samples w,r.t a positive sample.
            neg_sample_indexes = np.zeros((self.no_of_local_regions, self.no_of_neg_local_regions, 2), dtype=np.int32)
            # Each positive local region will have corresponding regions that act as negative samples to be
            # contrasted. For each positive sample, we pick the nearby no_of_neg_local_regions (5) local regions as
            # negative samples from both the images (x_a1_i, x_a2_i)

            if self.local_reg_size == 1:
                # local region size = 3x3
                # for local region at (0,0), define the negative samples co-ordinates accordingly
                neg_sample_indexes[0, :, :] = [[0, int(im_y / 2)], [int(im_x / 4), int(im_y / 4)],
                                               [int(im_x / 4), int(im_y / 2)], [int(im_x / 2), 0],
                                               [int(im_x / 2), int(im_y / 4)]]
                # similarly, define negative samples co-ordinates according to positive sample
                neg_sample_indexes[1, :, :] = [[0, 0], [0, im_y], [int(im_x / 4), int(im_y / 4)],
                                               [int(im_x / 4), int(3 * im_y / 4)], [int(im_x / 2), int(im_y / 2)]]
                neg_sample_indexes[2, :, :] = [[0, int(im_y / 2)], [int(im_x / 4), int(im_y / 2)],
                                               [int(im_x / 4), int(3 * im_y / 4)], [int(im_x / 2), int(3 * im_y / 4)],
                                               [int(im_x / 2), im_y]]
                neg_sample_indexes[3, :, :] = [[0, 0], [int(im_x / 4), int(im_y / 4)], [int(im_x / 2), int(im_y / 2)],
                                               [im_x, 0], [int(3 * im_x / 4), int(im_y / 4)]]
                neg_sample_indexes[4, :, :] = [[int(im_x / 4), int(im_y / 4)], [int(im_x / 4), int(3 * im_y / 4)],
                                               [int(3 * im_x / 4), int(im_y / 4)],
                                               [int(3 * im_x / 4), int(3 * im_y / 4)],
                                               [int(im_x / 2), 0]]
                neg_sample_indexes[5, :, :] = [[0, im_y], [int(im_x / 2), int(im_y / 2)], [im_x, im_y],
                                               [int(im_x / 4), int(3 * im_y / 4)],
                                               [int(3 * im_x / 4), int(3 * im_y / 4)]]
                neg_sample_indexes[6, :, :] = [[int(im_x / 2), 0], [int(im_x / 2), int(im_y / 4)],
                                               [im_x, int(im_y / 2)],
                                               [int(3 * im_x / 4), int(im_y / 4)], [int(3 * im_x / 4), int(im_y / 2)]]
                neg_sample_indexes[7, :, :] = [[int(im_x / 2), int(im_y / 2)], [im_x, 0], [im_x, im_y],
                                               [int(3 * im_x / 4), int(im_y / 4)],
                                               [int(3 * im_x / 4), int(3 * im_y / 4)]]
                neg_sample_indexes[8, :, :] = [[int(im_x / 2), int(3 * im_y / 4)], [int(im_x / 2), im_y],
                                               [im_x, int(im_y / 2)], [int(3 * im_x / 4), int(3 * im_y / 4)],
                                               [int(3 * im_x / 4), int(im_y / 2)]]
                neg_sample_indexes[9, :, :] = [[0, 0], [0, int(im_y / 2)], [int(im_x / 2), 0],
                                               [int(im_x / 2), int(im_y / 2)], [int(im_x / 4), int(3 * im_y / 4)]]
                neg_sample_indexes[10, :, :] = [[0, int(im_y / 2)], [0, im_y], [int(im_x / 4), int(im_y / 4)],
                                                [int(im_x / 2), int(im_y / 2)], [int(im_x / 2), im_y]]
                neg_sample_indexes[11, :, :] = [[int(im_x / 2), 0], [int(im_x / 2), int(im_y / 2)], [im_x, 0],
                                                [im_x, int(im_y / 2)], [int(3 * im_x / 4), int(3 * im_y / 4)]]
                neg_sample_indexes[12, :, :] = [[int(im_x / 2), int(im_y / 2)], [int(im_x / 2), im_y],
                                                [im_x, int(im_y / 2)], [im_x, im_y], [int(3 * im_x / 4), int(im_y / 4)]]
            else:
                # local region size = 1x1
                # for local region at (0,0), define the negative samples co-ordinates accordingly
                neg_sample_indexes[0, :, :] = [[0, int(im_y / 2)], [int(im_x / 2), 0], [int(im_x / 2), int(im_y / 2)],
                                               [int(im_x / 4), int(im_y / 4)], [int(im_x / 4), int(3 * im_y / 4)]]
                # similarly, define negative samples co-ordinates according to positive sample
                neg_sample_indexes[1, :, :] = [[0, 0], [0, im_y], [int(im_x / 2), int(im_y / 2)],
                                               [int(im_x / 4), int(im_y / 4)], [int(im_x / 4), int(3 * im_y / 4)]]
                neg_sample_indexes[2, :, :] = [[0, int(im_y / 2)], [int(im_x / 2), int(im_y / 2)],
                                               [int(im_x / 2), im_y],
                                               [int(3 * im_x / 4), int(im_y / 4)], [int(im_x / 4), int(3 * im_y / 4)]]
                neg_sample_indexes[3, :, :] = [[0, 0], [int(im_x / 2), int(im_y / 2)], [im_x, 0],
                                               [int(im_x / 4), int(im_y / 4)], [int(3 * im_x / 4), int(im_y / 4)]]
                neg_sample_indexes[4, :, :] = [[int(im_x / 4), int(im_y / 4)], [int(im_x / 4), int(3 * im_y / 4)],
                                               [int(3 * im_x / 4), int(im_y / 4)],
                                               [int(3 * im_x / 4), int(3 * im_y / 4)],
                                               [int(im_x / 2), 0]]
                neg_sample_indexes[5, :, :] = [[0, im_y], [int(im_x / 2), int(im_y / 2)], [im_x, im_y],
                                               [int(im_x / 4), int(3 * im_y / 4)],
                                               [int(3 * im_x / 4), int(3 * im_y / 4)]]
                neg_sample_indexes[6, :, :] = [[int(im_x / 2), 0], [int(im_x / 2), int(im_y / 2)],
                                               [im_x, int(im_y / 2)],
                                               [int(3 * im_x / 4), int(im_y / 4)], [int(im_x / 4), int(im_y / 4)]]
                neg_sample_indexes[7, :, :] = [[int(im_x / 2), int(im_y / 2)], [im_x, 0], [im_x, im_y],
                                               [int(3 * im_x / 4), int(im_y / 4)],
                                               [int(3 * im_x / 4), int(3 * im_y / 4)]]
                neg_sample_indexes[8, :, :] = [[int(im_x / 2), int(im_y / 2)], [int(im_x / 2), im_y],
                                               [im_x, int(im_y / 2)],
                                               [int(3 * im_x / 4), int(3 * im_y / 4)],
                                               [int(3 * im_x / 4), int(im_y / 4)]]
                neg_sample_indexes[9, :, :] = [[0, 0], [0, int(im_y / 2)], [int(im_x / 2), 0],
                                               [int(im_x / 2), int(im_y / 2)], [int(im_x / 4), int(3 * im_y / 4)]]
                neg_sample_indexes[10, :, :] = [[0, int(im_y / 2)], [0, im_y], [int(im_x / 2), int(im_y / 2)],
                                                [int(im_x / 2), im_y], [int(3 * im_x / 4), int(3 * im_y / 4)]]
                neg_sample_indexes[11, :, :] = [[int(im_x / 2), 0], [int(im_x / 2), int(im_y / 2)], [im_x, 0],
                                                [0, int(im_y / 2)], [int(im_x / 4), int(im_y / 4)]]
                neg_sample_indexes[12, :, :] = [[int(im_x / 2), int(im_y / 2)], [int(im_x / 2), im_y],
                                                [im_x, int(im_y / 2)], [im_x, im_y], [int(im_x / 4), int(3 * im_y / 4)]]

            return pos_sample_indexes, neg_sample_indexes

    def forward(self, y_fin):
        # Note: only L_r strategy (random sampling) from Chaintanya et. al.
        local_loss = 0
        # y_fin = y_fin_tmp
        # print('y_fin_local',y_fin)

        # loop over each image pair to iterate over all positive local regions within a feature map to calculate the
        # local contrastive loss
        for pos_index in range(0, self.bs, 2):

            # indexes of positive pair of samples (f_a1_i,f_a2_i) of input images (x_a1_i,x_a2_i) from the batch of
            # feature maps.
            num_i1 = np.arange(pos_index, pos_index + 1, dtype=np.int32)
            num_i2 = np.arange(pos_index + 1, pos_index + 2, dtype=np.int32)

            # gather required positive samples (f_a1_i,f_a2_i) of (x_a1_i,x_a2_i) for the numerator term
            x_num_i1 = y_fin[num_i1]  # tf.gather(y_fin, num_i1)
            x_num_i2 = y_fin[num_i2]  # tf.gather(y_fin, num_i2)
            # print('x_num_i1,x_num_i2',x_num_i1,x_num_i2)

            # if local region size is 3x3
            if self.local_reg_size == 1:
                # loop over all defined local regions within a feature map
                for local_pos_index in range(0, self.no_of_local_regions, 1):
                    # 'pos_index_num' is the positive local region index in feature map f_a1_i of image x_a1_i that
                    # contributes to the numerator term. Fetch x and y coordinates
                    tmp_idx_0 = torch.LongTensor([self.pos_sample_indexes[local_pos_index, 0],
                                                  self.pos_sample_indexes[local_pos_index, 0] + 1,
                                                  self.pos_sample_indexes[local_pos_index, 0] + 2])
                    tmp_idx_1 = torch.LongTensor([self.pos_sample_indexes[local_pos_index, 1],
                                                  self.pos_sample_indexes[local_pos_index, 1] + 1,
                                                  self.pos_sample_indexes[local_pos_index, 1] + 2])
                    x_num_tmp_i1 = torch.gather(x_num_i1, dim=1, index=tmp_idx_0)
                    x_num_tmp_i1 = torch.gather(x_num_tmp_i1, dim=2, index=tmp_idx_1)

                    x_n_i1_flat = torch.flatten(input=x_num_tmp_i1)
                    if self.wgt_en == 1:
                        dense = nn.Linear(in_features=len(x_n_i1_flat), out_features=128, bias=False)
                        x_w3_n_i1 = dense(x_n_i1_flat)
                        # tf.layers.dense(inputs=x_n_i1_flat, units=128, name='seg_pred', activation=None, use_bias=False, reuse=tf.AUTO_REUSE)
                    else:
                        x_w3_n_i1 = x_n_i1_flat

                    # corresponding positive local region index in feature map f_a2_i of image x_a2_i that
                    # contributes to the numerator term. Fetch x and y coordinates
                    x_num_tmp_i2 = torch.gather(x_num_i2, dim=1, index=tmp_idx_0)
                    x_num_tmp_i2 = torch.gather(x_num_tmp_i2, dim=2, index=tmp_idx_1)
                    x_n_i2_flat = torch.flatten(input=x_num_tmp_i2)
                    if self.wgt_en == 1:
                        dense = nn.Linear(in_features=len(x_n_i2_flat), out_features=128, bias=False)
                        x_w3_n_i2 = dense(x_n_i2_flat)
                    else:
                        x_w3_n_i2 = x_n_i2_flat

                    # calculate cosine similarity score for the pair of positive local regions with index
                    # 'pos_index_den' within the feature maps from images (x_a1_i,x_a2_i)

                    # loss for positive pairs of local regions in feature maps  (f_a1_i,f_a2_i) & (f_a2_i,f_a1_i) in
                    # (num_i1_loss,num_i2_loss)

                    # Numerator loss terms of local loss
                    num_i1_ss = self.cos_sim(x_w3_n_i1, x_w3_n_i2) / self.temp_fac
                    num_i2_ss = self.cos_sim(x_w3_n_i2, x_w3_n_i1) / self.temp_fac

                    # Negative local regions as per the chosen positive local region at index 'pos_index_den'
                    neg_samples_index_list = np.squeeze(self.neg_sample_indexes[local_pos_index])
                    no_of_neg_pts = len(neg_samples_index_list)

                    # Denominator loss terms of local loss
                    den_i1_ss, den_i2_ss = 0, 0

                    for local_neg_index in range(0, no_of_neg_pts, 1):
                        # negative local regions in feature map (f_a1_i) from image (x_a1_i)
                        neg_tmp_idx_0 = torch.LongTensor([neg_samples_index_list[local_neg_index, 0],
                                                          neg_samples_index_list[local_neg_index, 0] + 1,
                                                          neg_samples_index_list[local_neg_index, 0] + 2])
                        neg_tmp_idx_1 = torch.LongTensor([neg_samples_index_list[local_neg_index, 1],
                                                          neg_samples_index_list[local_neg_index, 1] + 1,
                                                          neg_samples_index_list[local_neg_index, 1] + 2])

                        x_den_tmp_i1 = torch.gather(x_num_i1, dim=1, index=neg_tmp_idx_0)
                        x_den_tmp_i1 = torch.gather(x_den_tmp_i1, dim=2, index=neg_tmp_idx_1)
                        x_d_i1_flat = torch.flatten(input=x_den_tmp_i1)
                        if self.wgt_en == 1:
                            dense = nn.Linear(in_features=len(x_d_i1_flat), out_features=128, bias=False)
                            x_w3_d_i1 = dense(x_d_i1_flat)
                        else:
                            x_w3_d_i1 = x_d_i1_flat

                        # negative local regions in feature map (f_a2_i) from image (x_a2_i)
                        x_den_tmp_i2 = torch.gather(x_num_i2, dim=1, index=neg_tmp_idx_0)
                        x_den_tmp_i2 = torch.gather(x_den_tmp_i2, dim=2, index=neg_tmp_idx_1)
                        x_d_i2_flat = torch.flatten(input=x_den_tmp_i2)
                        if self.wgt_en == 1:
                            dense = nn.Linear(in_features=len(x_d_i2_flat), out_features=128, bias=False)
                            x_w3_d_i2 = dense(x_d_i2_flat)
                        else:
                            x_w3_d_i2 = x_d_i2_flat

                        # cosine score b/w local region of feature map (f_a1_i) vs other local regions within the
                        # same feature map (f_a1_i)
                        den_i1_ss = den_i1_ss + torch.exp(self.cos_sim(x_w3_n_i1, x_w3_d_i1) / self.temp_fac)
                        # cosine score b/w local region of feature map (f_a1_i) vs other local regions from another
                        # feature map (f_a2_i)
                        den_i1_ss = den_i1_ss + torch.exp(self.cos_sim(x_w3_n_i1, x_w3_d_i2) / self.temp_fac)

                        # cosine score b/w local region of feature map (f_a2_i) vs other local regions within the
                        # same feature map (f_a2_i)
                        den_i2_ss = den_i2_ss + torch.exp(self.cos_sim(x_w3_n_i2, x_w3_d_i2) / self.temp_fac)
                        # cosine score b/w local region of feature map (f_a2_i) vs other local regions from another
                        # feature map (f_a1_i)
                        den_i2_ss = den_i2_ss + torch.exp(self.cos_sim(x_w3_n_i2, x_w3_d_i1) / self.temp_fac)

                    # local loss from feature map f_a1_i
                    num_i1_loss = -torch.log(torch.sum(torch.exp(num_i1_ss)) / (
                            torch.sum(torch.exp(num_i1_ss)) + torch.sum(den_i1_ss)))
                    # num_i1_loss=-tf.log(tf.exp(num_i1_ss)/(tf.exp(num_i1_ss)+tf.math.reduce_sum(den_i1_ss)))
                    local_loss = local_loss + num_i1_loss

                    # local loss from feature map f_a2_i
                    num_i2_loss = -torch.log(torch.sum(torch.exp(num_i2_ss)) / (
                            torch.sum(torch.exp(num_i2_ss)) + torch.sum(den_i2_ss)))
                    # num_i2_loss=-tf.log(tf.exp(num_i2_ss)/(tf.exp(num_i2_ss)+tf.math.reduce_sum(den_i2_ss)))
                    local_loss = local_loss + num_i2_loss

            # if local region size is 1x1
            else:
                # loop over all defined local regions within a feature map
                for local_pos_index in range(0, self.no_of_local_regions, 1):
                    # positive local region 'pos_index_den' in feature map from image x_a1_i
                    # fetch x and y coordinates
                    x_num_tmp_i1 = torch.gather(x_num_i1, dim=1,
                                                index=torch.LongTensor(self.pos_sample_indexes[local_pos_index, 0]))
                    x_num_tmp_i1 = torch.gather(x_num_tmp_i1, dim=1,
                                                index=torch.LongTensor(self.pos_sample_indexes[local_pos_index, 1]))

                    # corresponding positive local region 'pos_index_den' in feature map from image x_a2_i
                    # fetch x and y coordinates
                    x_num_tmp_i2 = torch.gather(x_num_i2, dim=1,
                                                index=torch.LongTensor(self.pos_sample_indexes[local_pos_index, 0]))
                    x_num_tmp_i2 = torch.gather(x_num_tmp_i2, dim=1,
                                                index=torch.LongTensor(self.pos_sample_indexes[local_pos_index, 1]))

                    # calculate cosine similarity score for the pair of positive local regions with index
                    # 'pos_index_den' within the feature maps from images (x_a1_i,x_a2_i)

                    # loss for positive pairs of local regions in feature maps  (f_a1_i,f_a2_i) & (f_a2_i,f_a1_i) in
                    # (num_i1_loss,num_i2_loss)

                    # Numerator loss terms of local loss
                    num_i1_ss = self.cos_sim(x_num_tmp_i1, x_num_tmp_i2) / self.temp_fac
                    num_i2_ss = self.cos_sim(x_num_tmp_i2, x_num_tmp_i1) / self.temp_fac

                    # Negative local regions as per the chosen positive local region at index 'pos_index_den'
                    neg_samples_index_list = np.squeeze(self.neg_sample_indexes[local_pos_index])
                    no_of_neg_pts = neg_samples_index_list.shape[0]

                    # Denominator loss terms of local loss
                    den_i1_ss, den_i2_ss = 0, 0

                    for local_neg_index in range(0, no_of_neg_pts, 1):
                        # negative local regions in feature map (f_a1_i) from image (x_a1_i)
                        x_den_tmp_i1 = torch.gather(x_num_i1, dim=1,
                                                    index=torch.LongTensor(neg_samples_index_list[local_neg_index, 0]))
                        x_den_tmp_i1 = torch.gather(x_den_tmp_i1, dim=1,
                                                    index=torch.LongTensor(neg_samples_index_list[local_neg_index, 1]))

                        # negative local regions in feature map (f_a2_i) from image (x_a2_i)
                        x_den_tmp_i2 = torch.gather(x_num_i2, dim=1,
                                                    index=torch.LongTensor(neg_samples_index_list[local_neg_index, 0]))
                        x_den_tmp_i2 = torch.gather(x_den_tmp_i2, dim=1,
                                                    index=torch.LongTensor(neg_samples_index_list[local_neg_index, 1]))

                        # cosine score b/w local region of feature map (f_a1_i) vs other local regions within the
                        # same feature map (f_a1_i)
                        den_i1_ss = den_i1_ss + torch.exp(self.cos_sim(x_num_tmp_i1, x_den_tmp_i1) / self.temp_fac)
                        # cosine score b/w local region of feature map (f_a1_i) vs other local regions from another
                        # feature map (f_a2_i)
                        den_i1_ss = den_i1_ss + torch.exp(self.cos_sim(x_num_tmp_i1, x_den_tmp_i2) / self.temp_fac)

                        # cosine score b/w local region of feature map (f_a2_i) vs other local regions within the
                        # same feature map (f_a2_i)
                        den_i2_ss = den_i2_ss + torch.exp(self.cos_sim(x_num_tmp_i2, x_den_tmp_i2) / self.temp_fac)
                        # cosine score b/w local region of feature map (f_a2_i) vs other local regions from another
                        # feature map (f_a1_i)
                        den_i2_ss = den_i2_ss + torch.exp(self.cos_sim(x_num_tmp_i2, x_den_tmp_i1) / self.temp_fac)

                    # local loss from feature map f_a1_i
                    num_i1_loss = -torch.log(torch.sum(torch.exp(num_i1_ss)) / (
                            torch.sum(torch.exp(num_i1_ss)) + torch.sum(den_i1_ss)))
                    local_loss = local_loss + num_i1_loss

                    # local loss from feature map f_a2_i
                    num_i2_loss = -torch.log(torch.sum(torch.exp(num_i2_ss)) / (
                            torch.sum(torch.exp(num_i2_ss)) + torch.sum(den_i2_ss)))
                    local_loss = local_loss + num_i2_loss

        local_loss = local_loss / self.no_of_local_regions
        net_local_loss = local_loss / self.bs

        return net_local_loss
