
# Reproducing: Communication-Efficient Learning of Deep Networks from Decentralized Data

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sWdbt_a3Dya9TQKTB2k5p-kRJWiznGsb)

## Introduction
In 2017, H. Brendan McMahan, et al. published Communication-Efficient Learning of Deep Networks from Decentralized Data. The paper describes the principle of Federated Learning and demonstrates a practical method of deep networks based on interative model averaging. 


This project is done by group 18 -- Stefan Hofman, Mirza Mrahorovic, and Yen-Lin Wu, as a part of TU Delft's CS4240 Deep Learning course. We reproduced the paper results, as well as randomized the data distribution before weight updates and added noise to weights before communication with clients. 



## Federated Learning

With increasing use of mobile devices, the data stored on them can be used to improve, for example, language models, voice recognition, and text entry. Federated Learning allows users to collectively reap the benefits of shared models trained from such data without the need to centrally store it. However, such data are privacy sensitive in nature.

In this project, we investigate this learning technique proposed by Google. It is termed *Federated Learning*, since the learning task is solved by a loose federation of participating devices (*clients*) coordinated by a central *server* -- <cite>[H.Brendan McMahan et al][1]</cite>. 
How the algorithm works is as follows. A central server shares its model weights with clients. Each client computes its own update on local training data and uploads it to the server that maintains the global model. The local training data set is never uploaded; instead, only the update is communicated for the global model. More formally, every client ($k$) trains on local data with SGD with a batch size $B$ to obtain a gradient estimate $g_k$.
\begin{align}
    w^{k} &\leftarrow w^{k} - \eta g_k
\end{align}
After $E$ epochs, the updated weights $w_k$ are sent to the server. The server then combines the weights of the clients to obtain a new model $w_{t+1}$. This is done by assigning a higher weight to clients with a larger fraction of data $\frac{n_k}{K}$ 
\begin{align}
    w_{t+1} &\leftarrow \frac{n_k}{K} \sum_{k=1}^{K}w^{k}_{t+1}   
\end{align}
This model is then redistributed back to the clients for further training.

![](https://i.imgur.com/W25lGiw.jpg)

Several questions arise here:
1. What happens to the convergence rate if the data is distributed in an uneven manner among users?
2. Would a simple weighted averaging of the models lead to faster convergence if the data is distributed in an uneven manner?
3. How much noise can the weight updates tolerate and what are the implications for data privacy?

In order to answer these questions, we adjusted the algorithm called *FederatedAveraging* that is introduced in the original paper. A series of 3 experiments are carried out to demonstrate its robustness to unbalanced and non-IID data distributions, as well as the ability to reduce the rounds of communication needed to train a deep network on decentralized data by orders of magnitude.

We start by replicating the results of the paper given the existing code in <cite>[GitHub][2]</cite>. These results are presented in *Replication*. Next, we will present and discuss the results of the three questions presented above in Section *Uneven Distribution*, *Weighted Uneven Distribution* and *Noise robustness*. 


## Replication
To validate the correctness of the code and see if any bugs are present, we first replicate the results from the paper. We attempted to reproduce the original paper results using CNN on MNIST database for both IID and non-IID cases. For clarity, we repeat the model updates (for the clients) and weighted algorithm (by the server):

***
<div style="background-color:#F5F5F5;font-family:'Lucida Console', monospace; font-size:13px">

for E epochs
&nbsp;for batch b $\in$ B
&nbsp;&nbsp;$w^{k} \leftarrow w^{k} - \eta \frac{1}{n_k} \nabla \sum_{i \in P_k}^{}L_i(w)$

for client k $\in$ K
&nbsp;$w_{t+1} \leftarrow \frac{n_k}{K} \sum_{k=1}^{K}w^{k}_{t+1}$
</div>

***
Here, the gradient estimate $g_k$ is written out: $g_k = \frac{1}{n_k} \nabla \sum_{i \in P_k}^{}L_i(w)$, where $L_i(w)$ is the loss function evaluated on data sample $i$ with model parameters $w$.

![](https://i.imgur.com/VcTL4Er.png)

## Uneven Distribution without Weighted Model Updates
To answer the first question, we must distribute the data in an uneven manner among users. Furthermore, the algorithm is adjusted such that the weighted averaging is taken out. In fact, the original code on Github did not have the weighted term and hence, no adjustments were needed.
Note that the model update is given by the following formulas:

***
<div style="background-color:#F5F5F5;font-family:'Lucida Console', monospace; font-size:13px">

for E epochs
&nbsp;for batch b $\in$ B
&nbsp;&nbsp;$w^{k} \leftarrow w^{k} - \eta \frac{1}{n_k} \nabla \sum_{i \in \color{#E21A1A}{P_k}}^{}L_i(w)$

for client k $\in$ K
&nbsp;$w_{t+1} \leftarrow \color{#E21A1A}{\frac{1}{K}} \sum_{k=1}^{K}w^{k}_{t+1}$
</div>

***

The set of data points $P_k$ that is used to estimate the gradient is denoted in red to emphasize the parameter that should be altered.

Now, the Python code that achieves this functionality is:

```python
def mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(
            all_idxs,
            random.randint(1,num_items),
            replace=False))
        print(len(dict_users[i]))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
```

The learning curve is shown below:

![](https://i.imgur.com/2PVH7la.png)

## Uneven Distribution with Weighted Model Updates
The question that arises now is: how can we improve the model convergence given uneven distribution of data? A simple solution would be to alter the algorithm such that the server averages the weights of the models (provided by the $k$ users) in a weighted manner. Note that the *Federated Averaging* algorithm already has this functionality according to original paper. However, the code on Github did not have this functionality and hence required adjustments. 

In equations, the red highlighted part is now investigated.
***
<div style="background-color:#F5F5F5;font-family:'Lucida Console', monospace; font-size:13px">

for E epochs
&nbsp;for batch b $\in$ B
&nbsp;&nbsp;$w^{k} \leftarrow w^{k} - \eta \frac{1}{n_k} \nabla \sum_{i \in \color{#E21A1A}{P_k}}^{}L_i(w)$

for client k $\in$ K
&nbsp;$w_{t+1} \leftarrow \color{#E21A1A}{\frac{n_k}{K}} \sum_{k=1}^{K}w^{k}_{t+1}$
</div>

***

Now, the Python code that achieves this functionality is: 

```python
def FedAvg(w, clients):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            tens = torch.mul(w[i][k], clients[i])
            w_avg[k] += tens
        w_avg[k] = torch.div(w_avg[k], sum(clients))
    return w_avg
```

The learning curve is shown below:

$\color{red}{INSERT IMAGE}$


## Noise Robustness
As last, we would like to know how much noise the weights can tolerate. The reason why we would like to investigate model convergence under noisy weight updates is because attacks could occur in which the original image of the client can be reconstructed based on the updated weights sent over to the server. If these weights are 'polluted' with noise, we expect that the derived image will also be polluted and thereby made useless to the attacker. 

In equations, we have adjusted the algorithm as follows:
***
<div style="background-color:#F5F5F5;font-family:'Lucida Console', monospace; font-size:13px">

for E epochs
&nbsp;for batch b $\in$ B
&nbsp;&nbsp;$w^{k} \leftarrow w^{k} - \eta \frac{1}{n_k} \nabla \sum_{i \in P_k}^{}L_i(w)$

for client k $\in$ K
&nbsp;$w_{t+1} \leftarrow \frac{n_k}{K} \sum_{k=1}^{K} \left(w^{k}_{t+1} \color{#E21A1A}{+ N\left(0, \sigma^2 \right)} \right)$
</div>

***

Now, the Python code that achieves this functionality is:

``` python
for layer in w:
    x = np.random.normal(0,sigma_squared,w[layer].size())
    x = np.reshape(x,w[layer].size())
    x = torch.from_numpy(x)
    w[layer] = w[layer]+x.cuda()
```

The learning curve for 100 epochs is presented below:
![](https://i.imgur.com/gFOXk2x.png)

Note that Gaussian noise with a variance of 0.2 seems to converge to a reasonable accuracy (95%). For Gaussian noise with a variance of 0.3, the model does not converge. The oscillations in the former is limited, but clearly present. Note that the oscillations in the latter model are too severe.

To investigate the implications for data privacy, attacks should be reconstructed. More specifically, based on the update


## Discussions and Conclusions


### Why our reproduction differs from the paper results?


### Limited computation power
Our biggest challenge while reproducing the results is the limited access to computation power. Especially, the cases where Epoch=20 takes 1 hour to run 10 communication rounds. It would have taken 100 hours to run a single E=20 case. The paper did not mention the specification of GPU used or the computation time.

[1]: https://arxiv.org/pdf/1602.05629.pdf
[2]: https://github.com/shaoxiongji/federated-learning


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
