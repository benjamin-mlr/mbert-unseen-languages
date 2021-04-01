from typing import Tuple
import torch
import torch.nn.functional as F
import numpy


from allennlp.nn.chu_liu_edmonds import decode_mst


def run_mst_decoding(batch_energy: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    heads = []
    head_tags = []
    for energy, length in zip(batch_energy.detach().cpu(), lengths):

        scores, tag_ids = energy.max(dim=0)
        scores[0, :] = 0
        instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

        # Find the labels which correspond to the edges in the max spanning tree.
        instance_head_tags = []
        for child, parent in enumerate(instance_heads):

            instance_head_tags.append(tag_ids[child, parent].item())
            # OLD: instance_head_tags.append(tag_ids[parent, child].item())
        # We don't care what the head or tag is for the root token, but by default it's
        # not necesarily the same in the batched vs unbatched case, which is annoying.
        # Here we'll just set them to zero.
        instance_heads[0] = 0
        instance_head_tags[0] = 0
        heads.append(instance_heads)
        head_tags.append(instance_head_tags)

    return torch.from_numpy(numpy.stack(heads)), torch.from_numpy(numpy.stack(head_tags))


def decode_mst_tree(pairwise_head_logits, attended_arcs, mask, lengths):


    normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(0, 3, 1, 2)

    minus_inf = -1e8
    minus_mask = (1 - mask.float()) * minus_inf
    attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
    # Shape (batch_size, sequence_length, sequence_length)

    normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)
    # normalized_arc_logits = attended_arcs.transpose(1, 2)

    batch_energy = torch.exp(normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits)

    return run_mst_decoding(batch_energy, lengths)

