
# Reproducing: Communication-Efficient Learning of Deep Networks from Decentralized Data

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sWdbt_a3Dya9TQKTB2k5p-kRJWiznGsb)

## Introduction
In 2017, H. Brendan McMahan, et al. published Communication-Efficient Learning of Deep Networks from Decentralized Data. The paper describes the principle of Federated Learning and demonstrates a practical method of deep networks based on interative model averaging. 


This project is done by group 18 -- Stefan Hofman, Mirza Mrahorovic, and Yen-Lin Wu, as a part of TU Delft's CS4240 Deep Learning course. We reproduced the paper results, as well as randomized the data distribution before weight updates and added noise to weights before communication with clients. 



## Federated Learning

With increasing use of mobile devices, the data stored on them can be used to improve, for example, language models, voice recognition, and text entry. Federated Learning allows users to collectively reap the benefits of shared models trained from such data without the need to centrally store it. However, such data are privacy sensitive in nature.

In this project, we investigate this learning technique proposed by Google. It is termed *Federated Learning*, since the learning task is solved by a loose federation of participating devices (*clients*) coordinated by a central *server* -- [H.Brendan McMahan et al](https://arxiv.org/pdf/1602.05629). 
How the algorithm works is as follows. A central server shares its model weights with clients. Each client computes its own update on local training data and uploads it to the server that maintains the global model. The local training data set is never uploaded; instead, only the update is communicated for the global model. More formally, every client (k) trains on local data with SGD with a batch size B to obtain a gradient estimate g<sub>k</sub>.

After E epochs, the updated weights w<sub>k</sub> are sent to the server. The server then combines the weights of the clients to obtain a new model w<sub>t+1</sub>. This is done by assigning a higher weight to clients with a larger fraction of data ![\frac{n_k}{K}].

This model is then re-distributed back to the clients for further training. The above explanation is visualized in the figure below.

![](https://i.imgur.com/W25lGiw.jpg)

Several questions arise here:
1. What happens to the convergence rate if the data is distributed in an uneven manner among users?
2. Would a simple weighted averaging of the models lead to faster convergence if the data is distributed in an uneven manner?
3. How much noise can the weight updates tolerate and what are the implications for data privacy?

In order to answer these questions, we adjusted the algorithm called *FederatedAveraging* that is introduced in the original paper. A series of 3 experiments are carried out to demonstrate its robustness to unbalanced and non-IID data distributions, as well as the ability to reduce the rounds of communication needed to train a deep network on decentralized data by orders of magnitude.

We start by replicating the results of the paper given the existing code in <cite>[GitHub][2]</cite>. These results are presented in *Replication*. Next, we will present and discuss the results of the three questions presented above in Section *Uneven Distribution*, *Weighted Uneven Distribution* and *Noise robustness*. 


## References
```
@article{mcmahan2016communication,
  title={Communication-efficient learning of deep networks from decentralized data},
  author={McMahan, H Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and others},
  journal={arXiv preprint arXiv:1602.05629},
  year={2016}
}

@article{ji2018learning,
  title={Learning Private Neural Language Modeling with Attentive Aggregation},
  author={Ji, Shaoxiong and Pan, Shirui and Long, Guodong and Li, Xue and Jiang, Jing and Huang, Zi},
  journal={arXiv preprint arXiv:1812.07108},
  year={2018}
}
```

Attentive Federated Learning [[Paper](https://arxiv.org/abs/1812.07108)] [[Code](https://github.com/shaoxiongji/fed-att)]

## Requirements
python 3.6  
pytorch>=0.4
