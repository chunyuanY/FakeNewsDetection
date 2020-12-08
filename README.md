# Paper of the source codes released:
Chunyuan Yuan, Qianwen Ma, Wei Zhou, Jizhong Han, Songlin Hu. Early Detection of Fake News by Utilizing the Credibility of News, Publishers, and Users Based on Weakly Supervised Learning, COLING 2020.

# Dependencies:
Gensim==3.7.2

Jieba==0.39

Scikit-learn==0.21.2

Pytorch==1.1.0


# Datasets
The main directory contains the directories of Weibo dataset and two Twitter datasets: twitter15 and twitter16. In each directory, there are:
- twitter15.train, twitter15.dev, and twitter15.test file: This files provide traing, development and test samples in a format like: 'source tweet ID \t source tweet content \t label'
  
- twitter15_graph.txt file: This file provides the source posts content of the trees in a format like: 'source tweet ID \t userID1:weight1 userID2:weight2 ...'  

These dastasets are preprocessed according to our requirement and original datasets can be available at https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0  (Twitter)  and http://alt.qcri.org/~wgao/data/rumdect.zip (Weibo).

If you want to preprocess the dataset by youself, you can use the word2vec used in our work. The pretrained word2vec can be available at https://drive.google.com/drive/folders/1IMOJCyolpYtoflEqQsj3jn5BYnaRhsiY?usp=sharing.


# Reproduce the experimental results:
1. create an empty directory: checkpoint/
2. run script run.py 


## Citation
If you find this code useful in your research, please cite our paper:
```
@inproceedings{yuan2020early,
  title={Early Detection of Fake News by Utilizing the Credibility of News, Publishers, and Users Based on Weakly Supervised Learning},
  author={Yuan, Chunyuan and Ma, Qianwen and Zhou, Wei and Han, Jizhong and Hu, Songlin},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={5444--5454},
  year={2020}
}
```

