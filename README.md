# Visual-Relationship-Detection-RWFN

This repository is the official implementation of the
paper "[Representing Prior Knowledge Using Randomly, Weighted Feature Networks for Visual Relationship Detection](https://arxiv.org/abs/2111.10686)"
. All source codes were implemented in Python 2.7.

## Abstract

The single-hidden-layer Randomly Weighted Feature Network (RWFN) introduced
by [Hong and Pavlic (2021)](https://arxiv.org/abs/2109.06663) was developed as an alternative to neural tensor network
approaches for relational learning tasks. Its relatively small footprint combined with the use of two randomized input
projections -- an insect-brain-inspired input representation and random Fourier features -- allow it to achieve rich
expressiveness for relational learning with relatively low training cost. In particular, when Hong and Pavlic compared
RWFN to Logic Tensor Networks (LTNs) for Semantic Image Interpretation (SII) tasks to extract structured semantic
descriptions from images, they showed that the RWFN integration of the two hidden, randomized representations better
captures relationships among inputs with a faster training process even though it uses far fewer learnable parameters.
In this paper, we use RWFNs to perform Visual Relationship Detection (VRD) tasks, which are more challenging SII tasks.
A zero-shot learning approach is used with RWFN that can exploit similarities with other seen relationships and
background knowledge -- expressed with logical constraints between subjects, relations, and objects -- to achieve the
ability to predict triples that do not appear in the training set. The experiments on the Visual Relationship Dataset to
compare the performance between RWFNs and LTNs, one of the leading Statistical Relational Learning frameworks, show that
RWFNs outperform LTNs for the predicate-detection task while using fewer number of adaptable parameters (1:56 ratio).
Furthermore, background knowledge represented by RWFNs can be used to alleviate the incompleteness of training sets even
though the space complexity of RWFNs is much smaller than LTNs (1:27 ratio).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Execution

We directly compare the performances between our method and LTNs using Visual Relationship Detection (VRD) dataset.
Original source codes of LTNs and dataset are
available [here](https://github.com/ivanDonadello/Visual-Relationship-Detection-LTN). All the details of best
hyperparameters for RWFNs are described in the paper.

### Training a grounded theory

To run a train use the following command:

```setup
python train.py
```

The trained grounded theories are saved in the ```models``` folder in the files; ```KB_nc_10000.ckpt``` (no constraints)
and
```KB_wc_10000.ckpt``` (with constraints) for LTNs and ```RWFN_KB_nc_10000.ckpt``` (no constraints) and
```RWFN_KB_wc_10000.ckpt``` (with constraints) for RWFNs. The number in the filename (```10000```) is a parameter in the
code to set the number of iterations.

### Evaluating the grounded theories

To run the evaluation use the following commands

```setup
python predicate_detection.py
python relationship_phrase_detection.py
```

Then, launch Matlab, move into the ```Visual-Relationship-Detection-master``` folder, execute the scripts
```predicate_detection_RWFN.m``` and ```relationship_phrase_detection_RWFN.m``` and see the results.

## Contributing

All content in this repository is licensed under the MIT license.

## Reference

You can cite our work:
```
@inproceedings{hong2022representing,
  title={Representing Prior Knowledge Using Randomly, Weighted Feature Networks for Visual Relationship Detection},
  author={Hong, Jinyung and Pavlic, Ted},
  booktitle={The First International Workshop on Combining Learning and Reasoning: Programming Languages, Formalisms, and Representations~(CLeaR) at the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
