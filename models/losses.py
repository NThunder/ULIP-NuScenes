'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

def get_unique_embeddings(text_labels, text_embed):
    labels_tensor = text_labels
    
    unique, idx, counts = torch.unique(labels_tensor, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=text_embed.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    
    # Извлекаем эмбеддинги

    # print("first_indicies:  ", text_labels, unique, first_indicies)
    
    return text_embed[first_indicies], idx

class ULIPWithImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        pc_embed = outputs['pc_embed']
        text_embed = outputs['text_embed']
        image_embed = outputs['image_embed']
        logit_scale = outputs['logit_scale']
        text_labels = outputs['text_labels']
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=pc_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # gather features from all GPUs
        pc_embed_all, text_embed_all, image_embed_all = \
            utils.all_gather_batch([pc_embed, text_embed, image_embed])
            

        unique_text_embed, indices = get_unique_embeddings(text_labels, text_embed)

        logits_per_pc_text = logit_scale * pc_embed_all @ unique_text_embed.t()  # [batch, global_num_unique]


        # cosine similarity as logits
        # logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        # logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
        logits_per_pc_image = logit_scale * pc_embed @ image_embed_all.t()
        logits_per_image_pc = logit_scale * image_embed @ pc_embed_all.t()

        loss =  F.cross_entropy(logits_per_pc_text, indices.to(pc_embed.device)) + \
                (F.cross_entropy(logits_per_pc_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred_text = torch.argmax(logits_per_pc_text, dim=-1)  # [local_batch_size]
            correct_text = (pred_text == indices).sum()
            pc_text_acc = 100 * correct_text / local_batch_size

            pred = torch.argmax(logits_per_pc_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_image_acc = 100 * correct / local_batch_size

        return {'loss': loss, 'ulip_loss': loss, 'ulip_pc_image_acc': pc_image_acc, 'ulip_pc_text_acc': pc_text_acc}
