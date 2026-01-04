# Neural Representations Directed Reading Program
Welcome to the Neural Representations DRP. Here are resources/ experiments relating to how computers represent abstract concepts. In particular, our focuses are on representational alignment using embedding geometry, and alignment metrics between representations.

DRP Symposium: January 14th 2026

poster: https://www.overleaf.com/read/qhkpdnxnhkcw#80a0bd
please message if you would like to speak/write something to show!

### Tuesday 18th November 2025
This week, the aim is to familiarise yourself with pytorch. Train a classification model on the MNIST dataset, which is a (if not the most) classic ML task. We can then extend this to try and simulate non-idealities when training on neuromorphic hardware.

Try to implement 
MINST Tutorial: https://nextjournal.com/gkoehler/pytorch-mnist

Extention 1: a "Comittee Machine": https://www.nature.com/articles/s41467-020-18098-0

Another extention related to what we have been talking about in the Friday sessions it to try and implement the simple linear regression task on cross-lingual data. Perhaps the biggest hurdle will be in downloading the embedding model/data, so there is some artifical data in XXXX(not done yet...). That being said, it would be nice to recreate some results so those of you who are more confident in using PyTorch can begin to implement some of the results in the experiments folder. 

Clone/fork this repository, set up a venv and install requirements 

repo with link to download real word embedding data.
https://github.com/artetxem/vecmap
https://github.com/facebookresearch/MUSE
easier to read implementation: https://github.com/n8686025/word2vec-translation-matrix/blob/master/jp-en-translation.py

### Tuesday 25th November 2025
Session is not on today. Continue with last week's tasks.

### Tuesday 2nd December 2025
Following our session last Friday, we are starting to work towards our new project. This will be the goal for the rest of our sessions this term. Check [here](specific_connections_in_embeddings) for development!

## Resources

### Embeddings

  - The Origins of Representation Manifolds in Large Language Models (Modell et al. 
    - https://www.arxiv.org/abs/2505.18235}
    - Paper formalising the structure of features of concepts. This is a paper by Alexander Modell, a research fellow here at Imperial.

        \item Google Explainer of Embeddings
        \url{https://developers.google.com/machine-learning/crash-course/embeddings/embedding-space}

        \item Topology of Word Embeddings: Singularities Reflect Polysemy \url{https://arxiv.org/pdf/2011.09413}\\
        Interesting paper about the topological properties of a "meaning space".
        

        \item NeurReps \url{https://www.neurreps.org}
        \\ 
        NeurReps is a workshop, where researchers publish findings specifically pertaining representations (ie. embeddings) in artificial and biological brains. With blog and further learning resources also.  

        \end{itemize}
    
    \item Bilingual Mapping Problem \begin{itemize}
        
        \item Exploiting Similarities among Languages for Machine Translation (Mikolov et al.) \url{https://arxiv.org/pdf/1309.4168}

        
        \item Learning principled bilingual mappings of word embeddings while preserving monolingual invariance (Mikel Artetxe et al.) \url{https://aclanthology.org/D16-1250.pdf} 
        
        More advanced topics in Bilingual Map Problem

        

        \item Bilingual Lexicon Induction for Low-Resource Languages using Graph Matching via Optimal Transport (Marchisio et al.) \url{https://aclanthology.org/2022.emnlp-main.164} 

        \item Beyond Offline Mapping: Learning Cross-lingual Word Embeddings through Context Anchoring (Ormazaba et al.)
        \url{https://aclanthology.org/2021.acl-long.506.pdf}
\url{https://aclanthology.org/2022.emnlp-main.164.pdf}

    \end{itemize}

    \item Representational Alignment \begin{itemize}

        \item The Platonic Representation Hypothesis (Isola et al.)\\\url{https://arxiv.org/abs/2405.07987}
        \\A key paper in the study of representational alignment, which posits that AI representations are converging to an objective reality.
        
        \item Emerging Cross-lingual Structure in Pretrained Language Models (Conneau et al.) \url{https://aclanthology.org/2020.acl-main.536.pdf?utm_source=chatgpt.com}

        \item UniReps \url{https://unireps.org}

        Leading workshop on the alignment of representations.

        \item Equivalence between representational similarity analysis, centered kernel alignment, and canonical correlations analysis (Williams et al.) \url{https://openreview.net/forum?id=zMdnnFasgC#discussion}
    \end{itemize}

    \item Computational Neuroscience \begin{itemize}
        \item \textit{Peter Gärdenfors} is a psychologist that pioneered ideas re: concepts in vector spaces in his books "Conceptual Spaces" and "The Geometry of Meaning".
        
        \item Weekly In-Person CompNeuro Lectures 
        \url{https://www.ucl.ac.uk/life-sciences/gatsby/news-and-events}

        These are weekly lectures hosted at the Gatsby institute in UCL. They are leaders in this field and I highly recommend keeping up with their work if you're interested in the field.
        
    \end{itemize}

### Representational Alignment

### Alignment Metrics

## Exercises
